import os
import sys
import random
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pytorch_warmup as warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import conch.open_clip_custom
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor
from open_clip import get_tokenizer, create_model_from_pretrained

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import (
    cls_acc,
    search_hp,
)


def _is_l2_normalized(x: torch.Tensor, dim=-1, atol=1e-2):
    n = x.norm(dim=dim)
    return torch.isfinite(n).all() and float((n - 1.0).abs().max().item()) < atol


def _check_finite(name, x: torch.Tensor):
    if not torch.isfinite(x).all():
        raise ValueError(f"[SANITY Tip] {name} contains NaN/Inf")


def _check_shape(name, x: torch.Tensor, expected):
    if tuple(x.shape) != tuple(expected):
        raise ValueError(
            f"[SANITY Tip] {name} shape {tuple(x.shape)} != expected {tuple(expected)}"
        )


def sanity_check_tip_adapter_any(
    cfg,
    backbone: str,
    clip_model,
    clip_weights: torch.Tensor,  # (D, C)
    cache_keys: torch.Tensor,  # (D, Ncache)
    cache_values: torch.Tensor,  # (Ncache, C)
    val_features: torch.Tensor,  # (Nval, D)
    val_labels: torch.Tensor,  # (Nval,)
    test_features: torch.Tensor = None,
    test_labels: torch.Tensor = None,
    do_tip_adapter_f: bool = True,
    batch_size_f: int = 8,
    verbose: bool = True,
):
    """
    Sanity check unifié Tip-Adapter / Tip-Adapter-F.
    - Vérifie shapes + dtypes + normalisation
    - Vérifie logits Zero-shot + Tip-Adapter
    - Vérifie backward/grads pour Tip-Adapter-F (sur un mini-batch synthétique)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 0) to device + fp32 matmul safe
    # -------------------------
    clip_weights = clip_weights.to(device=device, dtype=torch.float32)
    cache_keys = cache_keys.to(device=device, dtype=torch.float32)
    cache_values = cache_values.to(device=device, dtype=torch.float32)
    val_features = val_features.to(device=device, dtype=torch.float32)
    val_labels = val_labels.to(device=device)

    if test_features is not None:
        test_features = test_features.to(device=device, dtype=torch.float32)
    if test_labels is not None:
        test_labels = test_labels.to(device=device)

    # make sure norms are exactly what Tip-Adapter expects (safe)
    cache_keys = cache_keys / cache_keys.norm(dim=0, keepdim=True).clamp_min(1e-12)
    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True).clamp_min(
        1e-12
    )
    val_features = val_features / val_features.norm(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    if test_features is not None:
        test_features = test_features / test_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    # -------------------------
    # 1) dims
    # -------------------------
    Nval, D = val_features.shape
    D2, C = clip_weights.shape
    D3, Ncache = cache_keys.shape
    Ncache2, C2 = cache_values.shape

    if verbose:
        print("\n" + "=" * 90)
        print("🔍 SANITY CHECK Tip-Adapter (multi-backbone)")
        print("=" * 90)
        print("Backbone:", backbone)
        print(
            f"val_features : {tuple(val_features.shape)} {val_features.dtype} {val_features.device}"
        )
        print(
            f"clip_weights : {tuple(clip_weights.shape)} {clip_weights.dtype} {clip_weights.device}"
        )
        print(
            f"cache_keys   : {tuple(cache_keys.shape)} {cache_keys.dtype} {cache_keys.device}"
        )
        print(
            f"cache_values : {tuple(cache_values.shape)} {cache_values.dtype} {cache_values.device}"
        )

    # shape constraints
    assert D == D2 == D3, (
        f"[SANITY Tip] D mismatch: val D={D}, clip_weights D={D2}, cache_keys D={D3}"
    )
    assert C == C2, f"[SANITY Tip] C mismatch: clip_weights C={C}, cache_values C={C2}"
    assert Ncache == Ncache2, (
        f"[SANITY Tip] Ncache mismatch: cache_keys N={Ncache}, cache_values N={Ncache2}"
    )
    assert val_labels.numel() == Nval, (
        f"[SANITY Tip] val_labels size {val_labels.numel()} != Nval {Nval}"
    )

    # finite
    _check_finite("val_features", val_features)
    _check_finite("clip_weights", clip_weights)
    _check_finite("cache_keys", cache_keys)
    _check_finite("cache_values", cache_values)

    # normalized?
    if not _is_l2_normalized(val_features, dim=-1):
        raise ValueError(
            "[SANITY Tip] val_features not L2-normalized (did you forget feats/=norm?)"
        )
    if not _is_l2_normalized(clip_weights.t(), dim=-1):
        # clip_weights is (D,C) -> weights vectors are columns, so check on transpose
        raise ValueError("[SANITY Tip] clip_weights columns not L2-normalized")
    if not _is_l2_normalized(cache_keys.t(), dim=-1):
        # cache_keys is (D,N) -> keys vectors are columns
        raise ValueError("[SANITY Tip] cache_keys columns not L2-normalized")

    # -------------------------
    # 2) Zero-shot logits
    # -------------------------
    clip_logits = 100.0 * (val_features @ clip_weights)  # (Nval, C)
    _check_shape("clip_logits", clip_logits, (Nval, C))
    _check_finite("clip_logits", clip_logits)

    zs_pred = clip_logits.argmax(dim=1)
    zs_acc = float((zs_pred == val_labels).float().mean().item()) * 100.0

    if verbose:
        print(f"[SANITY Tip] Zero-shot val acc (quick): {zs_acc:.2f}%")

    # -------------------------
    # 3) Tip-Adapter logits
    # -------------------------
    beta = float(cfg.get("init_beta", 1.0))
    alpha = float(cfg.get("init_alpha", 1.0))

    affinity = val_features @ cache_keys  # (Nval, Ncache)
    _check_shape("affinity", affinity, (Nval, Ncache))
    _check_finite("affinity", affinity)

    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values  # (Nval, C)
    _check_shape("cache_logits", cache_logits, (Nval, C))
    _check_finite("cache_logits", cache_logits)

    tip_logits = clip_logits + cache_logits * alpha
    _check_shape("tip_logits", tip_logits, (Nval, C))
    _check_finite("tip_logits", tip_logits)

    tip_pred = tip_logits.argmax(dim=1)
    tip_acc = float((tip_pred == val_labels).float().mean().item()) * 100.0

    if verbose:
        print(f"[SANITY Tip] Tip-Adapter val acc (quick): {tip_acc:.2f}%")

    # -------------------------
    # 4) Optional: quick test set sanity
    # -------------------------
    if test_features is not None and test_labels is not None:
        Nt, Dt = test_features.shape
        assert Dt == D, f"[SANITY Tip] test D={Dt} != train/val D={D}"

        clip_logits_t = 100.0 * (test_features @ clip_weights)
        affinity_t = test_features @ cache_keys
        cache_logits_t = ((-1) * (beta - beta * affinity_t)).exp() @ cache_values
        tip_logits_t = clip_logits_t + cache_logits_t * alpha

        _check_shape("tip_logits_test", tip_logits_t, (Nt, C))
        _check_finite("tip_logits_test", tip_logits_t)

        acc_t = (
            float((tip_logits_t.argmax(dim=1) == test_labels).float().mean().item())
            * 100.0
        )
        if verbose:
            print(f"[SANITY Tip] Tip-Adapter test acc (quick): {acc_t:.2f}%")

    # -------------------------
    # 5) Tip-Adapter-F backward sanity
    # -------------------------
    if do_tip_adapter_f:
        # adapter: Linear(D -> Ncache) with weight (Ncache, D)
        adapter = nn.Linear(D, Ncache, bias=False).to(
            device=device, dtype=torch.float32
        )
        adapter.weight = nn.Parameter(cache_keys.t().contiguous())  # (Ncache, D)

        # mini-batch: take first few val samples
        b = min(batch_size_f, Nval)
        x = val_features[:b].detach()  # (b,D) already fp32
        y = val_labels[:b].detach()

        adapter.train()
        for p in adapter.parameters():
            p.grad = None

        aff = adapter(x)  # (b, Ncache)
        _check_shape("aff(adapter)", aff, (b, Ncache))

        cache_logits_b = ((-1) * (beta - beta * aff)).exp() @ cache_values  # (b,C)
        clip_logits_b = 100.0 * (x @ clip_weights)  # (b,C)
        tip_logits_b = clip_logits_b + cache_logits_b * alpha  # (b,C)

        loss = F.cross_entropy(tip_logits_b, y)
        if not torch.isfinite(loss):
            raise ValueError("[SANITY Tip] Tip-Adapter-F loss is NaN/Inf")

        loss.backward()

        # grads exist & finite only on adapter
        g = adapter.weight.grad
        if g is None:
            raise ValueError(
                "[SANITY Tip] adapter.weight.grad is None (no gradient flow?)"
            )
        if not torch.isfinite(g).all():
            raise ValueError("[SANITY Tip] adapter.weight.grad has NaN/Inf")
        gmean = float(g.detach().abs().mean().item())
        if gmean == 0.0:
            raise ValueError("[SANITY Tip] adapter grad mean abs is 0 (suspicious)")

        if verbose:
            print(
                f"[SANITY Tip] Tip-Adapter-F backward OK | loss={float(loss.item()):.6f} | mean|grad|={gmean:.6e}"
            )

    if verbose:
        print("✅ SANITY Tip-Adapter OK (shapes/dtypes/norms/logits/grads)")
        print("=" * 90 + "\n")

    return True


FINISH_MARKERS = [
    "Tip-Adapter-F's best test accuracy",  # ton exemple exact
    "After fine-tuning, Tip-Adapter-F's best test accuracy",  # variante possible
]


def run_is_completed(log_dir: Path) -> bool:
    """
    Retourne True si on détecte un signe clair que le run est terminé.
    Cherche dans un .out/.log (le plus récent) la présence d'un marker final.
    """
    if not log_dir.exists():
        return False

    # Cherche un fichier de log plausible (adapte si tu as un nom fixe)
    candidates = []
    for pat in ("*.out", "*.log", "*.txt"):
        candidates += list(log_dir.glob(pat))

    if not candidates:
        return False

    # Prend le plus récent
    log_path = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        # Lire seulement la fin (évite de charger 200MB en RAM)
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000), 0)  # ~200KB
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return False

    return any(m in tail for m in FINISH_MARKERS)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(cfg, shot, seed):
    dataset = cfg["dataset"]
    trainer = "Tip-Adapter"
    backbone = cfg["backbone"].replace("/", "")

    log_dir = Path(
        f"/gpfs/projects/acad/coalap/mdausort/ISBI_sup/output/"
        f"{dataset}/{trainer}/{backbone}/{shot}shots/seed{seed}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "log.txt"
    f = open(log_file, "w")

    # stdout/stderr -> fichier + stdout normal
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = sys.stdout

    print("========================================")
    print("Logging initialized")
    print(f"Log file : {log_file}")
    print("========================================")


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", help="settings of Tip-Adapter in yaml format"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for reproducibility"
    )
    parser.add_argument("--shots", type=int, default=1, help="Shots")
    args = parser.parse_args()

    return args


def load_backbone_and_preprocess(cfg):

    backbone = cfg["backbone"]

    if backbone == "Biomedclip":
        model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tok = get_tokenizer(model_id)
        return (
            model,
            preprocess,
            {"tokenizer": tok, "model_id": model_id, "kind": "openclip"},
        )

    elif backbone == "Quilt-B/16":
        model_id = "hf-hub:wisdomik/QuiltNet-B-16"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tok = get_tokenizer(model_id)
        return (
            model,
            preprocess,
            {"tokenizer": tok, "model_id": model_id, "kind": "openclip"},
        )

    elif backbone == "Quilt-B/32":
        model_id = "hf-hub:wisdomik/QuiltNet-B-32"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tok = get_tokenizer(model_id)
        return (
            model,
            preprocess,
            {"tokenizer": tok, "model_id": model_id, "kind": "openclip"},
        )

    elif backbone == "Conch":
        model, preprocess = conch.open_clip_custom.create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=os.environ.get("HF_TOKEN", None),
        )
        tok = conch.open_clip_custom.get_tokenizer()
        return model, preprocess, {"tokenizer": tok, "kind": "conch"}

    elif backbone == "PubMedCLIP-B/32":
        model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id)
        tok = CLIPTokenizerFast.from_pretrained(model_id)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        return model, preprocess, {"tokenizer": tok, "model_id": model_id, "kind": "hf"}

    elif backbone == "PLIP-B/32":
        model_id = "vinid/plip"
        model = CLIPModel.from_pretrained(model_id)
        tok = CLIPTokenizerFast.from_pretrained(model_id)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        return model, preprocess, {"tokenizer": tok, "model_id": model_id, "kind": "hf"}

    else:
        model_id = backbone
        clip_model, preprocess = clip.load(backbone)
        return clip_model, preprocess, {"kind": "clip"}


def clip_classifier(classnames, template, clip_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def biomedclip_classifier(classnames, templates, biomed_model, tokenizer, device=None):
    if device is None:
        device = next(biomed_model.parameters()).device

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]  # list[str]

            # ✅ open_clip tokenizer: pas de padding/truncation kwargs
            tok = tokenizer(texts)

            # open_clip peut renvoyer Tensor ou dict
            if isinstance(tok, dict):
                input_ids = tok.get("input_ids", list(tok.values())[0])
                input_ids = torch.as_tensor(input_ids).to(device)
            else:
                input_ids = torch.as_tensor(tok).to(device)

            # encode_text open_clip
            class_embeds = biomed_model.encode_text(input_ids)  # (n_templates, D)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)  # (D, n_cls)

    return clip_weights


def quilt_classifier(classnames, templates, quilt_model, tokenizer, device=None):
    if device is None:
        device = next(quilt_model.parameters()).device

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            tok = tokenizer(texts)
            # open_clip tokenizer: parfois dict, parfois tensor/list
            if isinstance(tok, dict):
                input_ids = tok.get("input_ids", list(tok.values())[0])
                input_ids = torch.as_tensor(input_ids).to(device)
            else:
                input_ids = torch.as_tensor(tok).to(device)

            # encode_text open_clip
            class_embeds = quilt_model.encode_text(input_ids)  # (n_templates, D)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)  # (D, n_cls)
    return clip_weights


def conch_classifier(classnames, templates, conch_model, tokenizer, device=None):
    if device is None:
        device = next(conch_model.parameters()).device

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            # --- Tokenize robust ---
            tok = None
            # 1) HF-style (works if tokenizer is a HF tokenizer)
            try:
                tok = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
            except TypeError:
                # 2) fallback: open_clip-style or custom
                tok = tokenizer(texts)

            if isinstance(tok, dict) or hasattr(tok, "input_ids"):
                # BatchEncoding or dict-like
                input_ids = tok["input_ids"] if isinstance(tok, dict) else tok.input_ids
                input_ids = input_ids.to(device)
            elif (
                isinstance(tok, (list, tuple))
                and len(tok) > 0
                and hasattr(tok[0], "ids")
            ):
                # list[tokenizers.Encoding]
                input_ids = torch.tensor(
                    [enc.ids for enc in tok], device=device, dtype=torch.long
                )
            else:
                # tensor / list[int] / list[list[int]]
                input_ids = torch.as_tensor(tok, device=device, dtype=torch.long)

            # --- Encode text ---
            class_embeds = conch_model.encode_text(input_ids)  # (n_templates, D)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)  # (D, n_cls)

    return clip_weights


def pubmedclip_classifier(classnames, templates, clip_model, tokenizer, device=None):
    if device is None:
        device = next(clip_model.parameters()).device

    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            tok = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}

            # HF CLIP: texte -> features projetées (dim CLIP)
            class_embeds = clip_model.get_text_features(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
            )  # (n_templates, D)

            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)  # (D, n_cls)
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache, shot):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = cfg["cache_dir"]
    keys_path = f"{cache_dir}/keys_{shot}shots.pt"
    values_path = f"{cache_dir}/values_{shot}shots.pt"

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []

                print(
                    "Augment Epoch: {:} / {:}".format(augment_idx, cfg["augment_epoch"])
                )
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.to(device)
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(device)
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys.cpu(), keys_path)
        torch.save(cache_values.cpu(), values_path)
    else:
        cache_keys = torch.load(keys_path, map_location=device)
        cache_values = torch.load(values_path, map_location=device)

    return cache_keys, cache_values


def build_cache_biomed(cfg, biomed_model, train_loader_cache, shot):

    cache_dir = cfg["cache_dir"]
    keys_path = f"{cache_dir}/keys_{shot}shots.pt"
    values_path = f"{cache_dir}/values_{shot}shots.pt"

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        biomed_model.eval()

        device = next(biomed_model.parameters()).device
        img_dtype = next(biomed_model.parameters()).dtype  # souvent fp16 si cuda

        with torch.no_grad():
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []

                print(f"Augment Epoch: {augment_idx} / {cfg['augment_epoch']}")
                for images, target in tqdm(train_loader_cache):
                    images = images.to(device, non_blocking=True)

                    # important: matcher le dtype du modèle (fp16 souvent)
                    image_features = biomed_model.encode_image(
                        images.to(dtype=img_dtype)
                    )
                    train_features.append(image_features)

                    if augment_idx == 0:
                        cache_values.append(target.to(device, non_blocking=True))

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        # moyenne sur augmentations
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)  # (N, D)
        cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)  # normalize
        cache_keys = cache_keys.permute(1, 0).contiguous()  # (D, N)

        # values one-hot
        targets = torch.cat(cache_values, dim=0)
        num_classes = int(targets.max().item()) + 1
        cache_values = F.one_hot(targets, num_classes=num_classes).half()  # (N, C)

        torch.save(cache_keys.cpu(), keys_path)
        torch.save(cache_values.cpu(), values_path)

    else:
        cache_keys = torch.load(keys_path, map_location=device)
        cache_values = torch.load(values_path, map_location=device)

    return cache_keys, cache_values


def build_cache_pubmedclip(cfg, clip_model, train_loader_cache, shot):

    cache_dir = cfg["cache_dir"]
    keys_path = f"{cache_dir}/keys_{shot}shots.pt"
    values_path = f"{cache_dir}/values_{shot}shots.pt"

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        # dtype vision HF (souvent float32)
        vision_dtype = next(clip_model.vision_model.parameters()).dtype
        device = next(clip_model.parameters()).device  # en pratique cuda

        with torch.no_grad():
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []

                print(f"Augment Epoch: {augment_idx} / {cfg['augment_epoch']}")
                for images, target in tqdm(train_loader_cache):
                    images = images.to(
                        device=device, dtype=vision_dtype, non_blocking=True
                    )

                    # HF CLIPModel
                    image_features = clip_model.get_image_features(pixel_values=images)
                    train_features.append(image_features)

                    if augment_idx == 0:
                        target = target.to(device=device, non_blocking=True)
                        cache_values.append(target)

                cache_keys.append(
                    torch.cat(train_features, dim=0).unsqueeze(0)
                )  # (1, N, D)

        # moyenne sur augmentations: (N, D)
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)

        # L2 norm + transpose vers (D, N) comme Tip-Adapter
        cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).contiguous()
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, keys_path)
        torch.save(cache_values, values_path)

    else:
        cache_keys = torch.load(keys_path, map_location="cpu")
        cache_values = torch.load(values_path, map_location="cpu")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not cfg["load_pre_feat"]:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(device), target.to(device)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg["cache_dir"] + "/" + split + "_f.pt")
        torch.save(labels, cfg["cache_dir"] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f.pt")
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l.pt")

    return features, labels


def pre_load_features_openclip(cfg, split, clip_model, loader):

    f_path = os.path.join(cfg["cache_dir"], f"{split}_f.pt")
    l_path = os.path.join(cfg["cache_dir"], f"{split}_l.pt")

    if not cfg["load_pre_feat"]:
        features, labels = [], []
        device = next(clip_model.parameters()).device
        img_dtype = next(clip_model.parameters()).dtype  # souvent fp16

        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(device=device, dtype=img_dtype, non_blocking=True)
                target = target.to(device=device, non_blocking=True)

                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)
        torch.save(features, f_path)
        torch.save(labels, l_path)

    else:
        # souvent tu veux charger CPU puis .to(device) plus tard
        features = torch.load(f_path, map_location=device)
        labels = torch.load(l_path, map_location=device)

    return features, labels


def _unwrap_pixel_values(images):
    # 1) Déjà un tensor batch (B,3,H,W)
    if torch.is_tensor(images):
        return images

    # 2) dict / BatchFeature / BatchEncoding -> extraire puis re-unwrap
    if isinstance(images, dict) and "pixel_values" in images:
        pv = images["pixel_values"]
        return _unwrap_pixel_values(pv)

    if hasattr(images, "pixel_values"):
        pv = images.pixel_values
        return _unwrap_pixel_values(pv)

    if hasattr(images, "data") and isinstance(images.data, dict) and "pixel_values" in images.data:
        pv = images.data["pixel_values"]
        return _unwrap_pixel_values(pv)

    # 3) liste/tuple: unwrap chaque élément puis stack
    if isinstance(images, (list, tuple)):
        pvs = []
        for it in images:
            pv = _unwrap_pixel_values(it)  # récursif
            pvs.append(pv)

        # squeeze batch=1 éventuel
        pvs2 = []
        for pv in pvs:
            if torch.is_tensor(pv) and pv.dim() == 4 and pv.size(0) == 1:
                pv = pv.squeeze(0)
            pvs2.append(pv)

        # si on a des tensors (3,H,W) on stack -> (B,3,H,W)
        if all(torch.is_tensor(pv) for pv in pvs2):
            return torch.stack(pvs2, dim=0)

        raise TypeError(f"List elements are not tensors after unwrap. Types: {[type(x) for x in pvs2]}")

    raise TypeError(f"Unsupported images type: {type(images)}")


def pre_load_features_hfclip(cfg, split, clip_model, loader):
    f_path = os.path.join(cfg["cache_dir"], f"{split}_f.pt")
    l_path = os.path.join(cfg["cache_dir"], f"{split}_l.pt")

    device = next(clip_model.parameters()).device
    vision_dtype = next(clip_model.vision_model.parameters()).dtype  # souvent fp32

    if not cfg["load_pre_feat"]:
        features, labels = [], []
        with torch.no_grad():
            for images, target in tqdm(loader):

                pixel_values = _unwrap_pixel_values(images)

                # DEBUG (à garder 1 run)
                print("RAW pixel_values:", type(pixel_values), getattr(pixel_values, "shape", None))

                # 1) si c'est une liste, on la convertit en tensor batch
                if isinstance(pixel_values, (list, tuple)):
                    pixel_values = _unwrap_pixel_values(pixel_values)

                # 2) squeeze toutes les dims "1" en trop au début (souvent (1,1,3,H,W) etc.)
                while torch.is_tensor(pixel_values) and pixel_values.dim() > 4 and pixel_values.size(0) == 1:
                    pixel_values = pixel_values.squeeze(0)

                # 3) cas fréquent: (B,1,3,H,W) -> (B,3,H,W)
                if torch.is_tensor(pixel_values) and pixel_values.dim() == 5 and pixel_values.size(1) == 1:
                    pixel_values = pixel_values.squeeze(1)

                # 4) si tu as (B,N,3,H,W) avec N>1, choisis une stratégie :
                if torch.is_tensor(pixel_values) and pixel_values.dim() == 5 and pixel_values.size(1) > 1:
                    # prendre la 1ère vue (simple)
                    pixel_values = pixel_values[:, 0]
                    # ou moyenne des features (plus propre) -> je te donne si tu veux

                # 5) image seule (3,H,W)
                if torch.is_tensor(pixel_values) and pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)

                print("FINAL pixel_values:", tuple(pixel_values.shape))

                assert pixel_values.dim() == 4, f"Expected 4D (B,3,H,W), got {tuple(pixel_values.shape)}"

                pixel_values = pixel_values.to(device=device, dtype=vision_dtype, non_blocking=True)
                image_features = clip_model.get_image_features(pixel_values=pixel_values)

                target = target.to(device=device, non_blocking=True)

                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)
        torch.save(features, f_path)
        torch.save(labels, l_path)
    else:
        features = torch.load(f_path, map_location=device)
        labels = torch.load(l_path, map_location=device)

    return features, labels


def encode_image_any(model, images, kind):
    device = next(model.parameters()).device
    images = images.to(device=device, non_blocking=True)

    if kind == "hf":
        vision_dtype = next(model.vision_model.parameters()).dtype
        return model.get_image_features(pixel_values=images.to(dtype=vision_dtype))

    elif kind in ["openclip", "conch", "biomedclip", "quilt"]:
        img_dtype = next(model.parameters()).dtype
        return model.encode_image(images.to(dtype=img_dtype))

    elif kind == "clip":
        # OpenAI CLIP encode_image attends float32 en général
        return model.encode_image(images.float())

    else:
        raise ValueError(f"Unknown kind={kind}")


def run_tip_adapter(
    cfg,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
):

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100.0 * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg["init_beta"], cfg["init_alpha"]

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        cfg, cache_keys, cache_values, val_features, val_labels, clip_weights
    )

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100.0 * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(
    cfg,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    clip_model,
    train_loader_F,
    extra,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable the cached keys to be learnable
    D, Ncache = cache_keys.shape
    adapter = nn.Linear(D, Ncache, bias=False).to(device=device, dtype=torch.float32)
    adapter.weight = nn.Parameter(cache_keys.t().contiguous())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], eps=1e-4)
    warmup_scheduler = warmup.LinearWarmup(optimizer, 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg["train_epoch"] * len(train_loader_F)
    )

    beta, alpha = cfg["init_beta"], cfg["init_alpha"]
    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0

    best_ckpt_path = os.path.join(
        cfg["cache_dir"], f"best_F_{cfg['shots']}shots.pt"
    )
    tmp_ckpt_path = os.path.join(
        cfg["cache_dir"], f"best_F_{cfg['shots']}shots.tmp.pt"
    )

    for train_idx in range(cfg["train_epoch"]):
        epoch = train_idx + 1

        # -------------------- Train --------------------
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []

        print(f"Train Epoch: {epoch} / {cfg['train_epoch']}")

        for images, target in tqdm(train_loader_F):
            images, target = images.to(device), target.to(device)

            with torch.no_grad():
                image_features = encode_image_any(clip_model, images, extra["kind"])
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            image_features = image_features.float()
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100.0 * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with warmup_scheduler.dampening():
                scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            "LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}".format(
                current_lr,
                correct_samples / all_samples,
                correct_samples,
                all_samples,
                sum(loss_list) / len(loss_list),
            )
        )

        # Eval
        adapter.eval()
        with torch.no_grad():
            affinity_val = adapter(val_features)
            cache_logits_val = ((-1) * (beta - beta * affinity_val)).exp() @ cache_values
            clip_logits_val = 100.0 * val_features @ clip_weights
            tip_logits_val = clip_logits_val + cache_logits_val * alpha
            val_acc = cls_acc(tip_logits_val, val_labels)

        print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(val_acc))

        if epoch <= 10:
            print(
                f"Early-stopping warmup epoch {epoch}/{10} "
                f"(no stopping, no patience counting)"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

                if os.path.exists(tmp_ckpt_path):
                    os.remove(tmp_ckpt_path)

                torch.save(adapter.weight.detach().cpu(), tmp_ckpt_path)
                os.replace(tmp_ckpt_path, best_ckpt_path)

            epochs_no_improve = 0
            continue

        # -------------------- Early stopping logic --------------------
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            print(
                f"New best validation accuracy: {best_val_acc:.2f} "
                f"at epoch {best_epoch}"
            )

            if os.path.exists(tmp_ckpt_path):
                os.remove(tmp_ckpt_path)

            torch.save(adapter.weight.detach().cpu(), tmp_ckpt_path)
            os.replace(tmp_ckpt_path, best_ckpt_path)

        else:
            epochs_no_improve += 1
            print(
                f"No improvement for {epochs_no_improve} epoch(s). "
                f"Best val accuracy: {best_val_acc:.2f} at epoch {best_epoch}"
            )

            if epochs_no_improve >= 10:
                print(
                    f"Early stopping triggered after {10} epochs "
                    f"without improvement."
                )
                break

    # -------------------- Reload best model --------------------
    if os.path.exists(best_ckpt_path):
        best_weight = torch.load(best_ckpt_path, map_location=device)
        adapter.weight = nn.Parameter(best_weight.to(device=device, dtype=torch.float32))
        print(
            f"**** Reloaded best adapter from epoch {best_epoch} "
            f"with val accuracy {best_val_acc:.2f}. ****\n"
        )
    else:
        print("Warning: no best checkpoint found, using current adapter.")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        clip_weights,
        adapter=adapter,
    )

    print("\n-------- Evaluating on the test set. --------")
    with torch.no_grad():
        affinity_test = adapter(test_features)
        cache_logits_test = ((-1) * (best_beta - best_beta * affinity_test)).exp() @ cache_values.to(
            affinity_test.dtype
        )
        clip_logits_test = 100.0 * test_features @ clip_weights
        tip_logits_test = clip_logits_test + cache_logits_test * best_alpha
        test_acc = cls_acc(tip_logits_test, test_labels)
    print("**** Tip-Adapter-F's final test accuracy: {:.2f}. ****\n".format(test_acc))

    if os.path.exists(best_ckpt_path):
        os.remove(best_ckpt_path)

    if os.path.exists(tmp_ckpt_path):
        os.remove(tmp_ckpt_path)


def main():

    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg["shots"] = args.shots
    dataset_log = cfg["dataset"]
    trainer_log = "Tip-Adapter"
    backbone_log = cfg["backbone"].replace("/", "")
    log_dir = Path(
        f"/gpfs/projects/acad/coalap/mdausort/ISBI_sup/output/"
        f"{dataset_log}/{trainer_log}/{backbone_log}/{args.shots}shots/seed{args.seed}"
    )

    if log_dir.exists() and run_is_completed(log_dir):
        print(f"[SKIP] already done: {log_dir}")
        return
    if log_dir.exists() and not run_is_completed(log_dir):
        print(f"[RE-RUN] directory exists but run incomplete: {log_dir}")

    setup_logging(cfg, args.shots, args.seed)

    cache_dir = os.path.join(
        "./caches",
        cfg["dataset"],
        backbone_log,
        f"{args.shots}shots",
        f"seed{args.seed}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # ----------------------- MODEL -----------------------
    clip_model, preprocess, extra = load_backbone_and_preprocess(cfg)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # ----------------------- Prepare dataset -----------------------
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])

    val_loader = build_data_loader(
        data_source=dataset.val,
        batch_size=64,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
    )
    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=64,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
    )

    if cfg["backbone"] == "Conch":
        train_tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=448,
                    scale=(0.5, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        train_tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.5, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    train_loader_cache = build_data_loader(
        data_source=dataset.train_x,
        batch_size=8,
        tfm=train_tranform,
        is_train=True,
        shuffle=False,
    )
    train_loader_F = build_data_loader(
        data_source=dataset.train_x,
        batch_size=8,
        tfm=train_tranform,
        is_train=True,
        shuffle=True,
    )

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    if cfg["backbone"] == "Biomedclip":
        clip_weights = biomedclip_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    elif cfg["backbone"] in ["Quilt-B/16", "Quilt-B/32"]:
        clip_weights = quilt_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    elif cfg["backbone"] == "Conch":
        clip_weights = conch_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        clip_weights = pubmedclip_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    else:
        clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    if cfg["backbone"] in ["Biomedclip", "Quilt-B/16", "Quilt-B/32", "Conch"]:
        cache_keys, cache_values = build_cache_biomed(
            cfg, clip_model, train_loader_cache, args.shots
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        cache_keys, cache_values = build_cache_pubmedclip(
            cfg, clip_model, train_loader_cache, args.shots
        )
    else:
        cache_keys, cache_values = build_cache_model(
            cfg, clip_model, train_loader_cache, args.shots
        )

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    if cfg["backbone"] in ["Biomedclip", "Quilt-B/16", "Quilt-B/32", "Conch"]:
        val_features, val_labels = pre_load_features_openclip(
            cfg, "val", clip_model, val_loader
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        val_features, val_labels = pre_load_features_hfclip(
            cfg, "val", clip_model, val_loader
        )
    else:
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg["backbone"] in ["Biomedclip", "Quilt-B/16", "Quilt-B/32", "Conch"]:
        test_features, test_labels = pre_load_features_openclip(
            cfg, "test", clip_model, test_loader
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        test_features, test_labels = pre_load_features_hfclip(
            cfg, "test", clip_model, test_loader
        )
    else:
        test_features, test_labels = pre_load_features(
            cfg, "test", clip_model, test_loader
        )

    clip_weights = clip_weights.to(device)
    cache_keys = cache_keys.to(device)
    cache_values = cache_values.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    cache_keys = cache_keys.float()
    cache_values = cache_values.float()
    val_features = val_features.float()
    test_features = test_features.float()
    clip_weights = clip_weights.float()

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
    )

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
        clip_model,
        train_loader_F,
        extra,
    )


if __name__ == "__main__":
    main()
