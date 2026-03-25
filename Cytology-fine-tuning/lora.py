import os 
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_warmup as warmup
from typing import Any, Callable
from utils import cls_acc, clip_classifier, get_function
import torch.nn.functional as F
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
)
from pathlib import Path
import sys


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_lora_params(model: nn.Module):
    lora_params = list(get_lora_parameters(model))
    total = sum(p.numel() for p in lora_params)
    trainable = sum(p.numel() for p in lora_params if p.requires_grad)
    return total, trainable, lora_params


def print_param_report(
    model_backbone: nn.Module, head: nn.Module | None = None, prefix="[PARAMS]"
):
    tot, tr = count_params(model_backbone)
    print(
        f"{prefix} backbone total={tot:,} | trainable={tr:,} | trainable%={100 * tr / max(tot, 1):.6f}%"
    )

    ltot, ltr, _ = count_lora_params(model_backbone)
    print(
        f"{prefix} LoRA total={ltot:,} | trainable={ltr:,} | trainable%={100 * ltr / max(ltot, 1):.6f}%"
    )

    if head is not None:
        htot, htr = count_params(head)
        print(f"{prefix} head total={htot:,} | trainable={htr:,}")
        print(f"{prefix} total trainable (LoRA+head) ~= {ltr + htr:,}")


def print_param_report_lora_text(model_vlm, prefix="[PARAMS]"):
    tot, tr = count_params(model_vlm)
    ltot, ltr, _ = count_lora_params(model_vlm)
    pct = 100.0 * tr / max(tot, 1)
    lpct = 100.0 * ltr / max(tot, 1)
    print(f"{prefix} total={tot:,} | trainable={tr:,} ({pct:.4f}%)")
    print(
        f"{prefix} LoRA = {ltr:,} trainable (LoRA tensors total={ltot:,}) ({lpct:.6f}% of total)"
    )


def get_logit_scale_from_model(model, default=1.0, exp_if_log=True):
    # OpenCLIP/CLIP: model.logit_scale is often a Parameter in log-space
    if hasattr(model, "logit_scale"):
        ls = model.logit_scale
        if torch.is_tensor(ls):
            # OpenCLIP stores logit_scale in log space; usually you want exp()
            return ls.exp() if exp_if_log else ls
        return torch.tensor(float(ls))
    return torch.tensor(float(default))


def sanity_check_run_lora(
    args,
    backbone: nn.Module,  # ton model (CLIP/openclip/timm/ViT/HF)
    model_vision: Callable,  # renvoyé par get_function(args, backbone)
    head: nn.Module,  # model_linear
    train_loader,
    *,
    n_steps=1,
    verbose=True,
):
    device = torch.device("cuda")
    backbone = backbone.to(device).train()
    head = head.to(device).train()

    # clear grads
    for p in backbone.parameters():
        p.grad = None
    for p in head.parameters():
        p.grad = None

    it = iter(train_loader)
    images, target, _ = next(it)

    # images doit être Tensor
    if not torch.is_tensor(images):
        raise TypeError(
            f"[SANITY run_lora] images is {type(images)} (expected torch.Tensor). "
            "Ton tfm doit renvoyer un Tensor (3,H,W), pas dict/BatchFeature/list."
        )

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        feats = model_vision(images)
        feats = _to_tensor(feats)
        feats = _pool_to_bd(feats)
        logits = head(feats)
        loss = F.cross_entropy(logits, target)

    if (
        logits.ndim != 2
        or logits.size(0) != images.size(0)
        or logits.size(1) != args.num_classes
    ):
        raise ValueError(
            f"[SANITY run_lora] logits shape {tuple(logits.shape)} != (B,{args.num_classes})"
        )

    if not torch.isfinite(loss):
        raise ValueError("[SANITY run_lora] loss is NaN/Inf")

    loss.backward()

    # grads LoRA
    lora_params = list(get_lora_parameters(backbone))
    if len(lora_params) == 0:
        raise ValueError(
            "[SANITY run_lora] No LoRA params found (apply_lora pas appliqué au bon module ?)"
        )

    n_lora_with_grad = sum(
        (p.grad is not None)
        and torch.isfinite(p.grad).all()
        and (p.grad.abs().sum() > 0)
        for p in lora_params
    )
    n_head_with_grad = sum(
        (p.grad is not None)
        and torch.isfinite(p.grad).all()
        and (p.grad.abs().sum() > 0)
        for p in head.parameters()
    )

    if verbose:
        print("\n" + "=" * 90)
        print("🔍 SANITY CHECK run_lora (image-only + head)")
        print("=" * 90)
        print("images:", tuple(images.shape), images.dtype, images.device)
        print("logits:", tuple(logits.shape), logits.dtype, logits.device)
        print("loss:", float(loss.item()))
        print(
            f"LoRA params: {len(lora_params)} | with nonzero grad: {n_lora_with_grad}"
        )
        print(
            f"Head params: {sum(1 for _ in head.parameters())} | with nonzero grad: {n_head_with_grad}"
        )

    if n_head_with_grad == 0:
        raise ValueError("[SANITY run_lora] No gradients on head (unexpected).")

    # Pour run_lora, LoRA peut être sur vision; si apply_lora met LoRA ailleurs, ça peut être 0.
    # On reste strict: on attend quand même des grads LoRA.
    if n_lora_with_grad == 0:
        raise ValueError(
            "[SANITY run_lora] No gradients on LoRA params. Causes typiques:\n"
            "- apply_lora appliqué sur un module différent de celui réellement utilisé par model_vision\n"
            "- model_vision pointe vers une copie/closure avant injection LoRA\n"
            "- forward est sous no_grad quelque part"
        )

    print("✅ SANITY run_lora PASSED\n")
    return True


def sanity_check_run_lora_text_v2(
    args,
    model_vlm,
    logit_scale,
    dataset,
    train_loader,
    *,
    verbose=True,
):
    device = torch.device("cuda")
    model_vlm = model_vlm.to(device)
    model_vlm.train()

    # IMPORTANT: récupérer functions APRÈS LoRA + to(device)
    model_vision, model_text, model_tokenizer = get_function(args, model_vlm)

    # logit_scale -> tensor
    if torch.is_tensor(logit_scale):
        ls = logit_scale.to(device=device, dtype=torch.float32)
    else:
        ls = torch.tensor(float(logit_scale), device=device, dtype=torch.float32)

    template = dataset.template[0]
    texts = [template.format(c.replace("_", " ")) for c in dataset.classnames]
    C = len(texts)

    # tokens une fois
    try:
        tok = model_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    except TypeError:
        tok = model_tokenizer(texts)

    # move tok to cuda (BatchEncoding/dict/tensor)
    if hasattr(tok, "to"):
        tok = tok.to(device)
    elif torch.is_tensor(tok):
        tok = tok.to(device)
    elif isinstance(tok, dict):
        tok = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tok.items()}

    # un batch
    it = iter(train_loader)
    batch = next(it)
    images, target, _ = unpack_batch(batch)

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    B = images.size(0)

    # clear grads
    for p in model_vlm.parameters():
        p.grad = None

    need_text_grad = args.encoder in ["text", "both"]
    need_vision_grad = args.encoder in ["vision", "both"]

    # text feats
    if need_text_grad:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_emb = model_text(tok)
            text_features = class_emb / class_emb.norm(dim=-1, keepdim=True)
    else:
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_emb = model_text(tok)
                text_features = class_emb / class_emb.norm(dim=-1, keepdim=True)

    # image feats
    if need_vision_grad:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = model_vision(images)
    else:
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = model_vision(images)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # align dtype/device
    text_features = text_features.to(
        dtype=image_features.dtype, device=image_features.device
    )
    ls_ = ls.to(dtype=image_features.dtype, device=image_features.device)

    logits = ls_ * (image_features @ text_features.t())
    assert logits.shape == (B, C), f"logits shape {tuple(logits.shape)} != {(B, C)}"
    loss = F.cross_entropy(logits, target)
    assert torch.isfinite(loss).all(), "loss is NaN/Inf"

    loss.backward()

    # check grads on LoRA params
    ltot, ltr, lora_params = count_lora_params(model_vlm)
    n_with = 0
    max_abs = 0.0
    for p in lora_params:
        if p.grad is None:
            continue
        n_with += 1
        max_abs = max(max_abs, float(p.grad.detach().abs().max().item()))

    if verbose:
        print("\n" + "=" * 90)
        print("🔍 SANITY CHECK LoRA-TEXT (v2)")
        print("=" * 90)
        print("encoder:", args.encoder)
        print("batch:", tuple(images.shape), "targets:", tuple(target.shape))
        print("logits:", tuple(logits.shape), "loss:", float(loss.item()))
        print(
            f"LoRA params tensors: {len(lora_params)} | with_grad: {n_with} | max|grad|={max_abs:.3e}"
        )
        print("=" * 90 + "\n")

    if len(lora_params) == 0:
        raise RuntimeError(
            "LoRA params list is empty -> apply_lora() didn’t modify the model used in forward."
        )
    if n_with == 0 or max_abs == 0.0:
        raise RuntimeError(
            "No non-zero grad on LoRA params. Causes typiques: "
            "get_function pointe vers mauvais sous-module, "
            "encoder mode met tout en no_grad, "
            "ou LoRA appliqué à un autre objet que celui utilisé dans le forward."
        )

    return True


def setup_logging_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = f
    sys.stderr = f


def get_run_dir(args) -> Path:
    # 1) priorité à output_dir si fourni
    out = getattr(args, "log_path", None)
    if out:
        run_dir = Path(out)
    else:
        run_dir = Path(args.results_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _device_of_callable(fn):
    # fn can be nn.Module or bound method (model.encode_text)
    if hasattr(fn, "parameters"):
        try:
            return next(fn.parameters()).device
        except StopIteration:
            pass
    mod = getattr(fn, "__self__", None)
    if mod is not None and hasattr(mod, "parameters"):
        try:
            return next(mod.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_tokens(tokens, device):
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if torch.is_tensor(tokens):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokens.items()
        }
    raise TypeError(f"Unsupported token type: {type(tokens)}")


def _to_tensor(x: Any) -> torch.Tensor:
    """Try to extract a tensor from common container outputs."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0:
        # often (features, ...) or (last_hidden_state, pooled, ...)
        for e in x:
            if torch.is_tensor(e):
                return e
        return _to_tensor(x[0])
    if isinstance(x, dict):
        # common keys
        for k in [
            "image_embeds",
            "text_embeds",
            "pooler_output",
            "last_hidden_state",
            "logits",
        ]:
            if k in x and torch.is_tensor(x[k]):
                return x[k]
        # fallback: first tensor value
        for v in x.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Could not extract tensor from output type={type(x)}")


def _pool_to_bd(x: torch.Tensor) -> torch.Tensor:
    """
    Convert possible shapes to [B, D]:
      - [B, D] -> OK
      - [B, T, D] -> take CLS token if plausible else mean pool over T
      - [B, C, H, W] -> global average pool -> [B, C]
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # Heuristic: token 0 is CLS for many ViTs
        # If T looks like tokens, pick CLS; otherwise mean pool.
        # Safer: take token 0 by default.
        return x[:, 0, :]
    if x.ndim == 4:
        return x.mean(dim=(-2, -1))
    raise ValueError(f"Unsupported feature tensor shape: {tuple(x.shape)}")


def infer_feature_dim(encode_image: Callable, loader, device=None) -> int:
    device = device or torch.device("cuda")
    for batch in loader:
        images, _, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            feats = encode_image(images)
        feats = _to_tensor(feats)
        feats = _pool_to_bd(feats)
        return int(feats.shape[-1])
    raise RuntimeError("Empty loader: cannot infer feature dim.")


def unpack_batch(batch):
    # dict (Dassl)
    if isinstance(batch, dict):
        images = batch["img"]
        target = batch["label"]
        path = batch.get("impath", batch.get("path", None))
        return images, target, path

    # tuple/list
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, target = batch
            return images, target, None
        if len(batch) >= 3:
            images, target, path = batch[0], batch[1], batch[2]
            return images, target, path

    raise TypeError(f"Unsupported batch type: {type(batch)}")


class ImageEncoderWithHead(nn.Module):
    def __init__(
        self, encode_image: Callable[[torch.Tensor], torch.Tensor], head: nn.Module
    ):
        super().__init__()
        # encode_image is a python callable closing over the backbone modules
        self.encode_image = encode_image
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image(images)
        feats = _to_tensor(feats)
        feats = _pool_to_bd(feats)
        return self.head(feats)


def evaluate_lora(args, model, loader):

    model.eval()

    acc = 0.0
    loss_epoch = 0.0
    tot_samples = 0
    with torch.no_grad():
        for batch in loader:
            images, target, _ = unpack_batch(batch)
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = F.cross_entropy(logits, target)
            loss_epoch += loss.item() * target.shape[0]
            acc += cls_acc(logits, target) * target.shape[0]
            tot_samples += target.shape[0]

    acc /= tot_samples
    loss_epoch /= tot_samples

    return acc


@torch.no_grad()
def evaluate_lora_dump(args, model, loader, out_csv: str | Path, model_tag: str):
    import time

    model.eval()
    rows = []

    total_inference_time = 0.0
    total_num_images = 0
    total_num_batches = 0
    warmup_batches = 5

    for batch_idx, batch in enumerate(loader):
        # support (images, target) ou (images, target, path, patient_id) ou dict
        if isinstance(batch, dict):
            images = batch["img"]
            target = batch["label"]
            paths = batch.get("path", [None] * len(target))
            patient_ids = batch.get("patient_id", [None] * len(target))
        else:
            if len(batch) == 2:
                images, target = batch
                paths = [None] * len(target)
                patient_ids = [None] * len(target)
            elif len(batch) == 3:
                images, target, paths = batch
                patient_ids = [None] * len(target)
            else:
                images, target, paths, patient_ids = batch

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        if batch_idx >= warmup_batches:
            batch_time = end_time - start_time
            batch_size = images.shape[0]

            total_inference_time += batch_time
            total_num_images += batch_size
            total_num_batches += 1

        probs = F.softmax(logits.float(), dim=1)
        pred = probs.argmax(dim=1)
        pmax = probs.max(dim=1).values

        for i in range(target.size(0)):
            row = {
                "model": model_tag,
                "path": paths[i] if isinstance(paths, (list, tuple)) else str(paths[i]),
                "patient_id": patient_ids[i] if isinstance(patient_ids, (list, tuple)) else patient_ids[i],
                "y_true": int(target[i].item()),
                "y_pred": int(pred[i].item()),
                "pmax": float(pmax[i].item()),
                "correct": int((pred[i] == target[i]).item()),
            }

            # ajouter les probabilités de chaque classe
            for c in range(probs.size(1)):
                row[f"prob_{c}"] = float(probs[i, c].item())

            rows.append(row)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    acc = df["correct"].mean() * 100.0

    if total_num_batches > 0 and total_num_images > 0:
        avg_time_per_batch = total_inference_time / total_num_batches
        avg_time_per_image = total_inference_time / total_num_images
        throughput = total_num_images / total_inference_time

        print("=== Inference time ===")
        print(f"Average inference time per batch: {avg_time_per_batch * 1000:.3f} ms")
        print(f"Average inference time per image: {avg_time_per_image * 1000:.3f} ms")
        print(f"Throughput: {throughput:.2f} images/s")
    else:
        print(
            f"Inference timing not computed: loader has <= {warmup_batches} batch(es) "
            "after warmup."
        )

    return acc, df


def evaluate_lora_text(args, model, loader, dataset):

    model_vision, model_text, model_tokenizer = get_function(args, model)

    model.eval()

    with torch.no_grad():
        template = dataset.template[0]
        texts = [
            template.format(classname.replace("_", " "))
            for classname in dataset.classnames
        ]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            tok = model_tokenizer(texts)

            # BatchEncoding / dict / tensor -> move to cuda
            if hasattr(tok, "to"):
                tok = tok.to("cuda")
            elif torch.is_tensor(tok):
                tok = tok.cuda()
            else:
                tok = {k: v.cuda() for k, v in tok.items()}

            class_embeddings = model_text(tok)

        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    loss_epoch = 0.0
    tot_samples = 0
    with torch.no_grad():
        for images, target, _ in loader:
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = model_vision(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            loss_epoch += loss.item() * target.shape[0]
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples
    loss_epoch /= tot_samples

    return acc


@torch.no_grad()
def evaluate_lora_text_dump(
    args, model, loader, dataset, out_csv, model_tag, logit_scale=None
):
    import time

    model_vision, model_text, model_tokenizer = get_function(args, model)
    model.eval()

    # --- text features ---
    template = dataset.template[0]
    texts = [template.format(c.replace("_", " ")) for c in dataset.classnames]

    text_device = _device_of_callable(model_text)
    vision_device = _device_of_callable(model_vision)

    tok = _move_tokens(model_tokenizer(texts), text_device)
    with torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(text_device.type == "cuda")
    ):
        class_embeddings = model_text(tok)

    class_embeddings = class_embeddings.float()
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    # --- pick logit_scale ---
    if logit_scale is None:
        if hasattr(model, "logit_scale") and torch.is_tensor(model.logit_scale):
            logit_scale = model.logit_scale.exp()
        else:
            logit_scale = 1.0

    if not torch.is_tensor(logit_scale):
        logit_scale = torch.tensor(float(logit_scale))

    rows = []

    # --- inference timing ---
    total_inference_time = 0.0
    total_num_images = 0
    total_num_batches = 0
    warmup_batches = min(5, max(0, len(loader) - 1))

    print(f"Inference timing warmup batches: {warmup_batches}")

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, dict):
            images = batch["img"]
            target = batch["label"]
            paths = batch.get("path", [None] * len(target))
            patient_ids = batch.get("patient_id", [None] * len(target))
        else:
            if len(batch) == 2:
                images, target = batch
                paths = [None] * len(target)
                patient_ids = [None] * len(target)
            elif len(batch) == 3:
                images, target, paths = batch
                patient_ids = [None] * len(target)
            else:
                images, target, paths, patient_ids = batch

        images = images.to(vision_device, non_blocking=True)
        target = target.to(vision_device, non_blocking=True)

        if vision_device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(vision_device.type == "cuda"),
        ):
            image_features = model_vision(images)

        if vision_device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        if batch_idx >= warmup_batches:
            batch_time = end_time - start_time
            batch_size = images.shape[0]

            total_inference_time += batch_time
            total_num_images += batch_size
            total_num_batches += 1

        image_features = image_features.float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tf = text_features.to(image_features.device)
        ls = logit_scale.to(device=image_features.device, dtype=image_features.dtype)

        logits = ls * (image_features @ tf.t())
        probs = F.softmax(logits.float(), dim=1)

        pred = probs.argmax(dim=1)
        pmax = probs.max(dim=1).values

        for i in range(target.size(0)):
            row = {
                "model": model_tag,
                "path": paths[i] if isinstance(paths, (list, tuple)) else str(paths[i]),
                "patient_id": patient_ids[i] if isinstance(patient_ids, (list, tuple)) else patient_ids[i],
                "y_true": int(target[i].item()),
                "y_pred": int(pred[i].item()),
                "pmax": float(pmax[i].item()),
                "correct": int((pred[i] == target[i]).item()),
            }

            for c in range(probs.size(1)):
                row[f"prob_{c}"] = float(probs[i, c].item())

            rows.append(row)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    acc = df["correct"].mean() * 100.0 if len(df) else 0.0

    # --- print inference speed ---
    if total_num_batches > 0 and total_num_images > 0:
        avg_time_per_batch = total_inference_time / total_num_batches
        avg_time_per_image = total_inference_time / total_num_images
        throughput = total_num_images / total_inference_time

        print("=== Inference speed ===")
        print(f"Average inference time per batch: {avg_time_per_batch * 1000:.3f} ms")
        print(f"Average inference time per image: {avg_time_per_image * 1000:.3f} ms")
        print(f"Throughput: {throughput:.2f} images/s")
    else:
        print("Inference timing not computed (not enough batches after warmup).")

    return acc, df


def run_lora(args, model, train_loader, val_loader, test_loader):

    device = torch.device("cuda")
    run_dir = get_run_dir(args)

    log_dir = Path(args.log_path) if getattr(args, "log_path", None) else run_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(args, "_logging_setup", False):
        setup_logging_to_file(log_dir / "log.txt")
        args._logging_setup = True

    # -------------------------- Define model --------------------------
    model = model.to(device)

    apply_lora(args, model)
    model = model.to(device)

    # IMPORTANT: after apply_lora
    model_vision, _, _ = get_function(args, model)

    num_features = infer_feature_dim(model_vision, train_loader, device=device)
    model_linear = nn.Linear(num_features, args.num_classes).to(device)

    mark_only_lora_as_trainable(model)

    trainable_parameters_ = list(get_lora_parameters(model))
    trainable_parameters_.extend(list(model_linear.parameters()))

    _model = ImageEncoderWithHead(model_vision, model_linear).to(device)

    print_param_report(model_backbone=model, head=model_linear, prefix="[PARAMS]")

    # -------------------------- Optimizer and scheduler --------------------------
    warmup_epochs = 10
    patience = 10

    n_iters_per_epoch = int(np.ceil(args.shots * args.num_classes / args.batch_size))
    total_iters = int(n_iters_per_epoch * args.n_epochs)
    warmup_period = int(n_iters_per_epoch * warmup_epochs)

    optimizer = torch.optim.AdamW(
        trainable_parameters_,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=1e-6
    )

    # -------------------------- training LoRA --------------------------
    scaler = torch.amp.GradScaler("cuda")
    count_iters = 0
    VALIDATION = True

    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    acc_val = None

    best_ckpt_path = run_dir / "best_model.pt"

    epoch = 0
    while count_iters < total_iters:
        epoch += 1
        _model.train()

        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0

        for images, target, _ in train_loader:
            images, target = images.cuda(), target.cuda()
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = _model(images)
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                scheduler.step()

            bs = target.shape[0]
            acc_train += cls_acc(output, target) * bs
            loss_epoch += float(loss.item()) * bs
            tot_samples += bs

            count_iters += 1
            if count_iters >= total_iters:
                break

        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{args.n_epochs} | "
            f"LR: {current_lr:.6f}, Acc: {acc_train:.4f}, Loss: {loss_epoch:.4f}"
        )

        # -------------------------- Validation + Early stopping --------------------------
        if VALIDATION:
            _model.eval()
            acc_val = evaluate_lora(args, _model, val_loader)
            print(f"**** Val accuracy: {acc_val:.2f}. ****")

            is_best = acc_val > best_val_acc

            if is_best:
                best_val_acc = acc_val
                best_epoch = epoch

                print(
                    f"New best model at epoch {epoch} "
                    f"with val accuracy {best_val_acc:.2f}"
                )

                tmp_ckpt_path = run_dir / "best_model.tmp.pt"
                if tmp_ckpt_path.exists():
                    tmp_ckpt_path.unlink()

                model_state_cpu = {
                    k: v.detach().cpu() for k, v in _model.state_dict().items()
                }

                ckpt = {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model_state_cpu,
                }

                torch.save(ckpt, tmp_ckpt_path)
                os.replace(tmp_ckpt_path, best_ckpt_path)

                del model_state_cpu
                del ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Warmup phase for early stopping
            if epoch <= 10:
                print(
                    f"Early-stopping warmup epoch {epoch}/{10} "
                    f"(no stopping, no patience counting)"
                )
                epochs_no_improve = 0

            else:
                if is_best:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(
                        f"No improvement for {epochs_no_improve} epoch(s). "
                        f"Best val accuracy = {best_val_acc:.2f} "
                        f"(epoch {best_epoch})"
                    )

                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping triggered after {patience} epochs "
                            f"without improvement."
                        )
                        break

            print()

    # -------------------------- Reload best model before test --------------------------
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        _model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Reloaded best model from epoch {checkpoint['epoch']} "
            f"with val accuracy {checkpoint['best_val_acc']:.2f}"
        )
    else:
        print("Warning: no best checkpoint found, using current model.")

    # -------------------------- Final test --------------------------
    pred_csv = run_dir / "predictions_test.csv"
    acc_test, _df = evaluate_lora_dump(
        args, _model, test_loader, pred_csv, model_tag="lora"
    )
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    # -------------------------- Save results --------------------------
    results = {
        "dataset": str(args.dataset),
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "shots": int(args.shots),
        "lr": float(args.lr),
        "r": int(args.r),
        "val_acc": float(best_val_acc if best_val_acc != float("-inf") else -1.0),
        "test_acc": float(acc_test),
        "encoder": str(args.encoder),
        "best_epoch": int(best_epoch) if best_epoch != -1 else -1,
        "early_stopped": bool(epochs_no_improve >= patience),
    }

    json_path = run_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if best_ckpt_path.exists():
        best_ckpt_path.unlink()
    return


@torch.no_grad()
def zeroshot_acc_stream(args, model_vlm, loader, textual_features, logit_scale):
    vision, _, _ = get_function(args, model_vlm)

    # vision peut être module ou bound-method
    dev = _device_of_callable(vision)

    tot = 0
    acc = 0.0

    # textual_features: [D, C] (comme ton clip_classifier)
    tf = textual_features.to(
        device=dev,
        dtype=torch.float16 if dev.type == "cuda" else torch.float32,
    )

    if torch.is_tensor(logit_scale):
        ls = logit_scale.to(device=dev, dtype=tf.dtype)
    else:
        ls = torch.tensor(float(logit_scale), device=dev, dtype=tf.dtype)

    for batch in loader:
        images, target, _ = unpack_batch(batch)
        images = images.to(dev, non_blocking=True)
        target = target.to(dev, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda")
        ):
            img = vision(images)
            img = img / img.norm(dim=-1, keepdim=True)

            logits = ls * (img @ tf)  # [B,C]

        acc += cls_acc(logits.float(), target) * target.size(0)
        tot += target.size(0)

    return acc / tot


def run_lora_text(
    args, model_vlm, dataset, train_loader, val_loader, test_loader
):
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    run_dir = get_run_dir(args)

    log_dir = Path(args.log_path) if getattr(args, "log_path", None) else run_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(args, "_logging_setup", False):
        setup_logging_to_file(log_dir / "log.txt")
        args._logging_setup = True

    device = torch.device("cuda")
    model_vlm = model_vlm.to(device)
    model_vlm.eval()

    logit_scale = get_logit_scale_from_model(model_vlm, default=1.0).to(
        device=device, dtype=torch.float32
    )
    logit_scale = logit_scale.detach()

    # -------------------------- Zero-shot part --------------------------
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(
        args, dataset.classnames, dataset.template, model_vlm
    )

    print("\nLoading visual features and labels from test set.")
    zs_acc = zeroshot_acc_stream(
        args, model_vlm, test_loader, textual_features, logit_scale
    )
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    # -------------------------- Define model --------------------------
    apply_lora(args, model_vlm)
    model_vlm = model_vlm.to(device)
    mark_only_lora_as_trainable(model_vlm)

    model_vision, model_text, model_tokenizer = get_function(args, model_vlm)

    print_param_report_lora_text(model_vlm, prefix="[PARAMS]")

    template = dataset.template[0]
    texts = [template.format(c.replace("_", " ")) for c in dataset.classnames]

    try:
        tok = model_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    except TypeError:
        tok = model_tokenizer(texts)

    if hasattr(tok, "to"):
        tok = tok.to(device)
    elif torch.is_tensor(tok):
        tok = tok.to(device)
    elif isinstance(tok, dict):
        tok = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tok.items()}

    # -------------------------- Optimizer and scheduler --------------------------
    warmup_epochs = 10
    patience = 10

    n_iters_per_epoch = int(np.ceil(args.shots * args.num_classes / args.batch_size))
    total_iters = int(n_iters_per_epoch * args.n_epochs)
    warmup_period = int(n_iters_per_epoch * warmup_epochs)

    optimizer = torch.optim.AdamW(
        get_lora_parameters(model_vlm),
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=1e-6
    )

    # -------------------------- Training --------------------------
    scaler = torch.amp.GradScaler("cuda")
    count_iters = 0
    VALIDATION = True

    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    acc_val = None
    best_ckpt_path = run_dir / "best_model.pt"
    epoch = 0
    early_stopped = False

    if args.encoder == "vision":
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                class_emb = model_text(tok)
                text_features_fixed = class_emb / class_emb.norm(dim=-1, keepdim=True)
        text_features_fixed = text_features_fixed.to(dtype=torch.float16)

    while count_iters < total_iters:
        epoch += 1
        model_vlm.train()

        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0

        for images, target, _ in train_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if args.encoder in ["text", "both"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    class_emb = model_text(tok)
                    text_features = class_emb / class_emb.norm(dim=-1, keepdim=True)
            else:
                text_features = text_features_fixed

            if args.encoder in ["vision", "both"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = model_vision(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = model_vision(images)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = text_features.to(
                dtype=image_features.dtype, device=image_features.device
            )

            ls_ = logit_scale.to(
                device=image_features.device, dtype=image_features.dtype
            )
            logits = ls_ * (image_features @ text_features.t())
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                scheduler.step()

            bs = target.size(0)
            acc_train += cls_acc(logits, target) * bs
            loss_epoch += float(loss.item()) * bs
            tot_samples += bs

            count_iters += 1
            if count_iters >= total_iters:
                break

        if tot_samples > 0:
            acc_train /= tot_samples
            loss_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{args.n_epochs} | "
            f"LR: {current_lr:.6f}, Acc: {acc_train:.4f}, Loss: {loss_epoch:.4f}"
        )

        # -------------------------- Validation + Early stopping --------------------------
        if VALIDATION:
            model_vlm.eval()
            acc_val = evaluate_lora_text(args, model_vlm, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****".format(acc_val))

            is_best = acc_val > best_val_acc

            if is_best:
                best_val_acc = acc_val
                best_epoch = epoch

                print(
                    f"New best model at epoch {epoch} "
                    f"with val accuracy {best_val_acc:.2f}"
                )

                tmp_ckpt_path = run_dir / "best_model.tmp.pt"
                if tmp_ckpt_path.exists():
                    tmp_ckpt_path.unlink()

                model_state_cpu = {
                    k: v.detach().cpu() for k, v in model_vlm.state_dict().items()
                }

                ckpt = {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model_state_cpu,
                }

                torch.save(ckpt, tmp_ckpt_path)
                os.replace(tmp_ckpt_path, best_ckpt_path)

                del model_state_cpu
                del ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Warmup phase for early stopping
            if epoch <= 10:
                print(
                    f"Early-stopping warmup epoch {epoch}/{10} "
                    f"(no stopping, no patience counting)"
                )
                epochs_no_improve = 0

            else:
                if is_best:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(
                        f"No improvement for {epochs_no_improve} epoch(s). "
                        f"Best val accuracy = {best_val_acc:.2f} "
                        f"(epoch {best_epoch})"
                    )

                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping triggered after {patience} epochs "
                            f"without improvement."
                        )
                        break

            print()

    # -------------------------- Reload best model --------------------------
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model_vlm.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Reloaded best model from epoch {checkpoint['epoch']} "
            f"with val accuracy {checkpoint['best_val_acc']:.2f}"
        )
    else:
        print("Warning: no best checkpoint found, using current model.")

    # -------------------------- Final test --------------------------
    pred_csv = run_dir / "predictions_test.csv"
    acc_test, _df = evaluate_lora_text_dump(
        args,
        model_vlm,
        test_loader,
        dataset,
        pred_csv,
        model_tag="lora",
        logit_scale=logit_scale,
    )
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    results = {
        "dataset": str(args.dataset),
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "shots": int(args.shots),
        "lr": float(args.lr),
        "r": int(args.r),
        "zs_acc": float(zs_acc),
        "val_acc": float(best_val_acc if best_val_acc != float("-inf") else -1.0),
        "test_acc": float(acc_test),
        "encoder": str(args.encoder),
        "best_epoch": int(best_epoch) if best_epoch != -1 else -1,
        "early_stopped": bool(early_stopped),
    }

    json_path = run_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if best_ckpt_path.exists():
        best_ckpt_path.unlink()

    return


def run_lora_features_extractor(args, model, train_loader, val_loader, test_loader):

    device = torch.device("cuda")
    run_dir = get_run_dir(args)

    log_dir = Path(args.log_path) if getattr(args, "log_path", None) else run_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(args, "_logging_setup", False):
        setup_logging_to_file(log_dir / "log.txt")
        args._logging_setup = True

    # -------------------------- Define model --------------------------
    model = model.to(device)

    # IMPORTANT: after apply_lora
    model_vision, _, _ = get_function(args, model)

    num_features = infer_feature_dim(model_vision, train_loader, device=device)
    model_linear = nn.Linear(num_features, args.num_classes).to(device)

    trainable_parameters_ = list(model_linear.parameters())

    _model = ImageEncoderWithHead(model_vision, model_linear).to(device)

    print_param_report(model_backbone=model, head=model_linear, prefix="[PARAMS]")

    # -------------------------- Optimizer and scheduler --------------------------
    warmup_epochs = 10
    patience = 10

    n_iters_per_epoch = int(np.ceil(args.shots * args.num_classes / args.batch_size))
    total_iters = int(n_iters_per_epoch * args.n_epochs)
    warmup_period = int(n_iters_per_epoch * warmup_epochs)

    optimizer = torch.optim.AdamW(
        trainable_parameters_,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=1e-6
    )

    # -------------------------- training LoRA --------------------------
    scaler = torch.amp.GradScaler("cuda")
    count_iters = 0
    VALIDATION = True

    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    acc_val = None

    best_ckpt_path = run_dir / "best_model.pt"

    epoch = 0
    while count_iters < total_iters:
        epoch += 1
        _model.train()

        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0

        for images, target, _ in train_loader:
            images, target = images.cuda(), target.cuda()
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = _model(images)
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                scheduler.step()

            bs = target.shape[0]
            acc_train += cls_acc(output, target) * bs
            loss_epoch += float(loss.item()) * bs
            tot_samples += bs

            count_iters += 1
            if count_iters >= total_iters:
                break

        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{args.n_epochs} | "
            f"LR: {current_lr:.6f}, Acc: {acc_train:.4f}, Loss: {loss_epoch:.4f}"
        )

        # -------------------------- Validation + Early stopping --------------------------
        if VALIDATION:
            _model.eval()
            acc_val = evaluate_lora(args, _model, val_loader)
            print(f"**** Val accuracy: {acc_val:.2f}. ****")

            is_best = acc_val > best_val_acc

            if is_best:
                best_val_acc = acc_val
                best_epoch = epoch

                print(
                    f"New best model at epoch {epoch} "
                    f"with val accuracy {best_val_acc:.2f}"
                )

                tmp_ckpt_path = run_dir / "best_model.tmp.pt"
                if tmp_ckpt_path.exists():
                    tmp_ckpt_path.unlink()

                model_state_cpu = {
                    k: v.detach().cpu() for k, v in _model.state_dict().items()
                }

                ckpt = {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model_state_cpu,
                }

                torch.save(ckpt, tmp_ckpt_path)
                os.replace(tmp_ckpt_path, best_ckpt_path)

                del model_state_cpu
                del ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Warmup phase for early stopping
            if epoch <= 10:
                print(
                    f"Early-stopping warmup epoch {epoch}/{10} "
                    f"(no stopping, no patience counting)"
                )
                epochs_no_improve = 0

            else:
                if is_best:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(
                        f"No improvement for {epochs_no_improve} epoch(s). "
                        f"Best val accuracy = {best_val_acc:.2f} "
                        f"(epoch {best_epoch})"
                    )

                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping triggered after {patience} epochs "
                            f"without improvement."
                        )
                        break

            print()

    # -------------------------- Reload best model before test --------------------------
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        _model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Reloaded best model from epoch {checkpoint['epoch']} "
            f"with val accuracy {checkpoint['best_val_acc']:.2f}"
        )
    else:
        print("Warning: no best checkpoint found, using current model.")

    # -------------------------- Final test --------------------------
    pred_csv = run_dir / "predictions_test.csv"
    acc_test, _df = evaluate_lora_dump(
        args, _model, test_loader, pred_csv, model_tag="feat_extract"
    )
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    # -------------------------- Save results --------------------------
    results = {
        "dataset": str(args.dataset),
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "shots": int(args.shots),
        "lr": float(args.lr),
        "r": int(args.r),
        "val_acc": float(best_val_acc if best_val_acc != float("-inf") else -1.0),
        "test_acc": float(acc_test),
        "encoder": str(args.encoder),
        "best_epoch": int(best_epoch) if best_epoch != -1 else -1,
        "early_stopped": bool(epochs_no_improve >= patience),
    }

    json_path = run_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if best_ckpt_path.exists():
        best_ckpt_path.unlink()

    return
