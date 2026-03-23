"""
Task Residual Tuning
by Tao Yu (yutao666@mail.ustc.edu.cn)
Oct 4, 2022
"""

import os.path as osp
import torch
import re
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import csv
from pathlib import Path
from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from open_clip import get_tokenizer, create_model_from_pretrained
from transformers import CLIPModel, CLIPTokenizerFast
import conch.open_clip_custom


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    "APACC": "a photo of a {}.",
    "BCFC": "a photo of a {}.",
    "BloodMNIST": "a photo of a {}.",
    "BMCD": "a photo of a {}.",
    "BMT": "a photo of a {}.",
    "EuroSAT": "a photo of a {}.",
    "FNAC2019": "a photo of a {}.",
    "Herlev": "a photo of a {}.",
    "HiCervix": "a photo of a {}.",
    "MLCC": "a photo of a {}.",
    "SiPakMed": "a photo of a {}.",
}


def _infer_path_from_batch(batch, i):
    if "impath" in batch:
        return batch["impath"][i]
    return ""


def infer_patient_id_from_path(path: str) -> str:
    if not path:
        return ""
    stem = Path(path).stem  # filename sans extension

    # Option A (souvent vraie): patient = avant le 1er underscore
    # ex: P001_23_patch_004.png -> P001
    pid = stem.split("_")[0]
    if pid:
        return pid

    # Option B: patient = avant le 1er dash
    pid = stem.split("-")[0]
    if pid:
        return pid

    # Option C: regex fallback (ex: patient123)
    m = re.search(r"(patient\d+|p\d+)", stem.lower())
    if m:
        return m.group(1)

    return stem


def sanity_check_taskres(
    trainer,
    batch=None,
    n_debug_classes=3,
    eps=1e-6,
    require_logits_change=True,
    verbose=True,
):

    model = trainer.model
    m = model.module if hasattr(model, "module") else model

    # ------------------------------------------------------------------
    # 0) Récupérer batch
    # ------------------------------------------------------------------
    if batch is None:
        batch = next(iter(trainer.train_loader_x))
    img, y = trainer.parse_batch_train(batch)

    if verbose:
        print("\n" + "=" * 100)
        print("🔍 SANITY CHECK TaskRes (all backbones)")
        print("=" * 100)
        print("Model:", type(m).__name__)
        print("img:", img.shape, img.device, img.dtype, "| y:", y.shape)

    # ------------------------------------------------------------------
    # 1) Vérifier presence prompt_learner + buffers attendus
    # ------------------------------------------------------------------
    assert hasattr(m, "prompt_learner"), "Model has no prompt_learner"
    pl = m.prompt_learner
    assert hasattr(pl, "base_text_features"), (
        "prompt_learner missing base_text_features buffer"
    )
    assert hasattr(pl, "text_feature_residuals"), (
        "prompt_learner missing text_feature_residuals param"
    )
    assert hasattr(pl, "alpha"), "prompt_learner missing alpha"

    base = pl.base_text_features
    res = pl.text_feature_residuals
    alpha = float(pl.alpha)

    assert base.dim() == 2, (
        f"base_text_features should be (C,D), got {tuple(base.shape)}"
    )
    assert res.shape == base.shape, (
        f"residuals shape {tuple(res.shape)} != base {tuple(base.shape)}"
    )

    n_cls, feat_dim = base.shape
    if verbose:
        print(
            f"[1] base_text_features: {tuple(base.shape)} dtype={base.dtype} device={base.device}"
        )
        print(
            f"[1] residuals:        {tuple(res.shape)} dtype={res.dtype} device={res.device} requires_grad={res.requires_grad}"
        )
        print(f"[1] alpha: {alpha}")

    # ------------------------------------------------------------------
    # 2) Trainables: idéalement uniquement text_feature_residuals (+ éventuellement logit_scale)
    # ------------------------------------------------------------------
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    if verbose:
        print(f"[2] Trainable params ({len(trainable)}):")
        for n, p in trainable:
            print("  ✔", n, tuple(p.shape), p.dtype, p.device)

    bad_trainable = []
    for n, p in trainable:
        # on tolère logit_scale si tu veux le fine-tuner (mais en TaskRes original c'est gelé)
        if "prompt_learner.text_feature_residuals" in n:
            continue
        if n.endswith("logit_scale"):
            # toléré, mais on le signale
            continue
        bad_trainable.append(n)

    assert len(bad_trainable) == 0, f"Unexpected trainable params: {bad_trainable[:10]}"

    # ------------------------------------------------------------------
    # 3) prompt_learner() doit être exactement base + alpha*res
    # ------------------------------------------------------------------
    pl.eval()
    tf = pl()  # (C,D)
    assert tf.shape == (n_cls, feat_dim)
    assert torch.isfinite(tf).all(), "prompt_learner() produced NaN/Inf"

    # check exact formula
    tf_ref = base + alpha * res
    max_err = (tf - tf_ref).abs().max().item()
    if verbose:
        print(f"[3] prompt_learner() OK | max|tf-(base+a*res)| = {max_err:.3e}")
    assert max_err < 1e-5, "prompt_learner() != base + alpha*residuals (unexpected)"

    # ------------------------------------------------------------------
    # 4) Forward pass: logits BxC, finite
    # ------------------------------------------------------------------
    m.train()
    model.zero_grad(set_to_none=True)

    # cast image si le backbone l'exige (vision dtype)
    # -> on reprend ta logique "try image.type(self.dtype) else float"
    # ici: on laisse tel quel, ton forward gère
    out = model(img)
    assert out.dim() == 2 and out.size(0) == img.size(0) and out.size(1) == n_cls, (
        f"logits shape expected (B,C)=({img.size(0)},{n_cls}), got {tuple(out.shape)}"
    )

    finite = torch.isfinite(out).all().item()
    if verbose:
        print(
            f"[4] logits: {tuple(out.shape)} dtype={out.dtype} device={out.device} finite={finite}"
        )
        print(f"    logits min/max: {out.min().item():.4g} / {out.max().item():.4g}")

    if not finite:
        # diagnostic: logit_scale exp
        ls = getattr(m, "logit_scale", None)
        if ls is not None:
            try:
                lse = ls.detach().float().exp().item()
                print(f"[4][DIAG] logit_scale.exp() = {lse}  (may overflow in fp16)")
            except Exception as e:
                print("[4][DIAG] cannot exp(logit_scale):", e)
        raise FloatingPointError("Non-finite logits detected")

    # ------------------------------------------------------------------
    # 5) Loss + backward: grads doivent passer dans residuals
    # ------------------------------------------------------------------
    loss = F.cross_entropy(out.float(), y)  # float() pour stabilité
    if verbose:
        print(f"[5] loss: {float(loss.item()):.6f}")

    loss.backward()

    # grads residuals
    g_res = pl.text_feature_residuals.grad
    assert g_res is not None, "No gradient on text_feature_residuals"
    assert torch.isfinite(g_res).all(), "Gradient on residuals has NaN/Inf"
    g_mean = g_res.abs().mean().item()
    g_max = g_res.abs().max().item()
    if verbose:
        print(f"[5] grad residuals absmean/max: {g_mean:.3e} / {g_max:.3e}")
    assert g_max > 0, "Residuals gradient is zero (unexpected)"

    # grads ailleurs (doivent être None pour params gelés)
    bad_grads = []
    for n, p in m.named_parameters():
        if (not p.requires_grad) and (p.grad is not None):
            bad_grads.append(n)
    assert len(bad_grads) == 0, f"Frozen params received grads: {bad_grads[:10]}"

    # ------------------------------------------------------------------
    # 6) Sensibilité: perturber residuals -> logits doivent changer
    # ------------------------------------------------------------------
    with torch.no_grad():
        out1 = model(img).float()
        # petite perturbation
        pl.text_feature_residuals.add_(
            0.01 * torch.randn_like(pl.text_feature_residuals)
        )
        out2 = model(img).float()
        delta = (out2 - out1).abs().mean().item()

    if verbose:
        print(f"[6] mean|logits(after perturb)-logits| = {delta:.6e}")

    if require_logits_change:
        assert delta > 0, (
            "Logits did not change after residual perturbation (unexpected)"
        )

    # ------------------------------------------------------------------
    # 7) Check base_text_features immuable (buffer, no grad)
    # ------------------------------------------------------------------
    assert pl.base_text_features.requires_grad is False, (
        "base_text_features should be a buffer (requires_grad=False)"
    )
    if pl.base_text_features.grad is not None:
        raise AssertionError("base_text_features has grad (should not)")

    # ------------------------------------------------------------------
    # 8) mini check sur quelques classes (optionnel)
    # ------------------------------------------------------------------
    if verbose and n_cls >= 2:
        a = tf[0:1]
        b = tf[1:2]
        cos = F.cosine_similarity(a, b).item()
        print(f"[8] cos(tf[0], tf[1]) = {cos:.4f}")

    if verbose:
        print("✅ SANITY CHECK TaskRes: OK")
        print("=" * 100 + "\n")

    return True


def _as_input_ids(tok_out):
    # dict / BatchEncoding -> input_ids
    if isinstance(tok_out, dict):
        ids = tok_out.get("input_ids", list(tok_out.values())[0])
    else:
        ids = tok_out
    # BatchEncoding attr
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    # tensor
    return torch.as_tensor(ids, dtype=torch.long)


def _extract_input_ids_any(tok, device, max_len=77, pad_id=0):
    """
    Return torch.LongTensor (B, L) padded/truncated to max_len.
    Handles: Tensor, dict, HF BatchEncoding, tokenizers.Encoding, python lists.
    """
    # 1) already tensor
    if torch.is_tensor(tok):
        ids = tok
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device=device)
        return _pad_trunc(ids.long(), max_len=max_len, pad_id=pad_id)

    # 2) HF BatchEncoding (transformers)
    # BatchEncoding has .data (dict) + supports __getitem__
    if hasattr(tok, "data") and isinstance(tok.data, dict):
        tok = tok.data  # convert to plain dict

    # 3) dict-like
    if isinstance(tok, dict):
        ids = tok.get("input_ids", None)
        if ids is None:
            # fallback: first entry
            ids = tok[next(iter(tok.keys()))]
        return _extract_input_ids_any(ids, device, max_len=max_len, pad_id=pad_id)

    # 4) tokenizers.Encoding (from `tokenizers` lib)
    # Can appear if tokenizer returns Encoding or list[Encoding]
    if hasattr(tok, "ids") and isinstance(tok.ids, list):
        # single Encoding -> make batch of 1
        row = tok.ids[:max_len]
        if len(row) < max_len:
            row = row + [int(pad_id)] * (max_len - len(row))
        return torch.tensor([row], dtype=torch.long, device=device)

    # list[Encoding]
    if isinstance(tok, (list, tuple)) and len(tok) > 0 and hasattr(tok[0], "ids"):
        padded = []
        for enc in tok:
            row = enc.ids[:max_len]
            if len(row) < max_len:
                row = row + [int(pad_id)] * (max_len - len(row))
            padded.append(row)
        return torch.tensor(padded, dtype=torch.long, device=device)

    # 5) python lists (list[int] or list[list[int]])
    if isinstance(tok, list):
        if len(tok) == 0:
            raise ValueError("Empty tokenizer output")
        if isinstance(tok[0], int):
            row = tok[:max_len]
            if len(row) < max_len:
                row = row + [int(pad_id)] * (max_len - len(row))
            return torch.tensor([row], dtype=torch.long, device=device)
        if isinstance(tok[0], (list, tuple)):
            padded = []
            for row in tok:
                row = list(row)[:max_len]
                if len(row) < max_len:
                    row = row + [int(pad_id)] * (max_len - len(row))
                padded.append(row)
            return torch.tensor(padded, dtype=torch.long, device=device)

    raise TypeError(f"Unsupported tokenizer output type: {type(tok)}")


def _pad_trunc(ids: torch.Tensor, max_len=77, pad_id=0):
    """ids: (B, L) -> (B, max_len)"""
    if ids.size(1) > max_len:
        return ids[:, :max_len]
    if ids.size(1) < max_len:
        pad = ids.new_full((ids.size(0), max_len - ids.size(1)), int(pad_id))
        return torch.cat([ids, pad], dim=1)
    return ids


def _vision_dtype_from_module(vision: torch.nn.Module) -> torch.dtype:
    # OpenAI CLIP / ViT-like : conv1 existe
    if hasattr(vision, "conv1") and hasattr(vision.conv1, "weight"):
        return vision.conv1.weight.dtype
    # fallback robuste
    return next(vision.parameters()).dtype


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    if backbone_name == "BiomedCLIP":
        model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        model, _ = create_model_from_pretrained(model_id, device="cpu")

        return model

    if backbone_name == "Quilt-B/32":
        model_id = "hf-hub:wisdomik/QuiltNet-B-32"

        model, _ = create_model_from_pretrained(model_id, device="cpu")

        return model

    if backbone_name == "Quilt-B/16":
        model_id = "hf-hub:wisdomik/QuiltNet-B-16"

        model, _ = create_model_from_pretrained(model_id, device="cpu")

        return model

    if backbone_name == "PubMedCLIP-B/32":
        model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"

        model = CLIPModel.from_pretrained(model_id)

        return model

    if backbone_name == "PLIP-B/32":
        model_id = "vinid/plip"

        model = CLIPModel.from_pretrained(model_id)

        return model

    if backbone_name == "Conch":
        model, preprocess = conch.open_clip_custom.create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token="REMOVED_TOKEN",
        )

        return model

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


# TaskRes(-Text)
class TaskResLearner(nn.Module):
    def __init__(self, cfg, base_text_features):
        super().__init__()
        self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        return (
            self.base_text_features + self.alpha * self.text_feature_residuals
        )  # t + a * x


# # TaskRes-Image
# class TaskResLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model, base_text_features):
#         super().__init__()
#         self.device = clip_model.dtype
#         # feat_dim = base_text_features.size(-1)
#         self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
#         print(">> DCT scale factor: ", self.alpha)
#         self.register_buffer("base_text_features", base_text_features)
#         self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features[0:1]))

#     def forward(self):
#         # print(self.base_text_features.dtype, self.text_feature_residuals.dtype)
#         return self.base_text_features, self.alpha * self.text_feature_residuals


def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

    dataset = cfg.DATASET.NAME

    TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize(
                [template.format(text) for template in TEMPLATES]
            )  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(
                    text_encoder(embeddings.cuda(), tokens.cuda())
                )  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


def _get_enhanced_base_text_features(
    cfg, classnames, clip_model, text_encoder, pretraiend_model
):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

        pretrained_text_projection = torch.load(pretraiend_model)

        state_dict = text_encoder.state_dict()
        state_dict["text_projection"] = pretrained_text_projection["state_dict"][
            "weight"
        ].t()
        text_encoder.load_state_dict(state_dict)
        print(">> Pretrained text encoder loaded!")
        params = pretrained_text_projection["state_dict"]["weight"].size(
            0
        ) * pretrained_text_projection["state_dict"]["weight"].size(1)
        print(">> Text projection parameters: ", params)
        print(pretrained_text_projection["state_dict"].keys())

    dataset = cfg.DATASET.NAME
    TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize(
                [template.format(text) for template in TEMPLATES]
            )  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(
                    text_encoder(embeddings.cuda(), tokens.cuda())
                )  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


def _get_base_text_features_biomed(cfg, classnames, biomed_model, tokenizer):
    device = next(biomed_model.parameters()).device
    dtype = next(biomed_model.parameters()).dtype

    dataset = cfg.DATASET.NAME
    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for name in classnames:
            name = name.replace("_", " ")
            prompts_list = [template.format(name) for template in TEMPLATES]

            # open_clip tokenizer: parfois accepte juste tokenizer(list_of_str)
            try:
                tok = tokenizer(
                    prompts_list,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
            except TypeError:
                tok = tokenizer(prompts_list)

            input_ids = _as_input_ids(tok).to(device)  # (n_templates, L)

            feats = biomed_model.encode_text(input_ids).to(
                dtype=dtype
            )  # (n_templates, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            text_embeddings.append(feats.mean(dim=0))  # (D,)

        text_embeddings = torch.stack(text_embeddings, dim=0)  # (n_cls, D)

    return text_embeddings


def _get_base_text_features_quilt(cfg, classnames, quilt_model, tokenizer):

    device = next(quilt_model.parameters()).device
    dtype = next(quilt_model.parameters()).dtype

    dataset = cfg.DATASET.NAME
    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]  # même pattern que toi

    with torch.no_grad():
        text_embeddings = []
        for name in classnames:
            name = name.replace("_", " ")

            prompts_list = [template.format(name) for template in TEMPLATES]

            tok = tokenizer(prompts_list)

            # open_clip tokenizer peut retourner dict ou tensor
            if isinstance(tok, dict):
                input_ids = tok.get("input_ids", list(tok.values())[0])
            else:
                input_ids = tok

            if hasattr(input_ids, "input_ids"):
                input_ids = input_ids.input_ids

            input_ids = torch.as_tensor(input_ids).to(device)  # (n_templates, L)

            feats = quilt_model.encode_text(input_ids).to(
                dtype=dtype
            )  # (n_templates, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            text_embeddings.append(feats.mean(dim=0))  # (D,)

        text_embeddings = torch.stack(text_embeddings, dim=0)  # (n_cls, D)

    return text_embeddings


def _get_base_text_features_conch(cfg, classnames, conch_model, tokenizer):
    device = next(conch_model.parameters()).device

    dataset = cfg.DATASET.NAME
    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]

    # si conch expose un pad_id, sinon 0
    pad_id = int(getattr(conch_model, "pad_id", 0) or 0)

    with torch.no_grad():
        text_embeddings = []
        for name in classnames:
            name = name.replace("_", " ")
            prompts_list = [template.format(name) for template in TEMPLATES]

            tok = tokenizer(
                prompts_list,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            input_ids = _extract_input_ids_any(
                tok, device, max_len=77, pad_id=pad_id
            )  # (n_templates, 77)
            feats = conch_model.encode_text(input_ids)

            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_embeddings.append(feats.mean(dim=0))

        text_embeddings = torch.stack(text_embeddings, dim=0)  # (n_cls, D)

    return text_embeddings


def _get_base_text_features_pubmedclip(cfg, classnames, clip_model, tokenizer):
    device = next(clip_model.parameters()).device
    dataset = cfg.DATASET.NAME

    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]  # comme toi

    with torch.no_grad():
        text_embeddings = []

        for name in classnames:
            name = name.replace("_", " ")
            prompts_list = [template.format(name) for template in TEMPLATES]

            tok = tokenizer(
                prompts_list,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}

            feats = clip_model.get_text_features(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
            )  # (n_templates, D)

            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_embeddings.append(feats.mean(dim=0))  # (D,)
        text_embeddings = torch.stack(text_embeddings, dim=0)  # (n_cls, D)
    return text_embeddings


# TaskRes by Tao Yu, Oct 4, 2022
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype  # float16
        text_encoder = TextEncoder(clip_model)
        if cfg.TRAINER.TaskRes.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features = _get_base_text_features(
                cfg, classnames, clip_model, text_encoder
            )
        else:
            print(">> Use enhanced base!")
            base_text_features = _get_enhanced_base_text_features(
                cfg,
                classnames,
                clip_model,
                text_encoder,
                cfg.TRAINER.TaskRes.ENHANCED_BASE,
            )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)

    def forward(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:
            image_features = self.image_encoder(image.float())

        # TaskRes-Text
        text_features = self.prompt_learner()

        # # TaskRes-Image
        # text_features, image_res = self.prompt_learner()
        # image_features += image_res

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomBiomedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model

        self.image_encoder = self.clip_model.encode_image
        self.logit_scale = getattr(self.clip_model, "logit_scale", None)
        self.vision = self.clip_model.visual

        self.text = self.clip_model.text
        self.word_embeddings = self.text.transformer.embeddings.word_embeddings

        base_text_features = _get_base_text_features_biomed(
            cfg, classnames, clip_model, tokenizer
        )

        # if cfg.TRAINER.TaskRes.ENHANCED_BASE == "none":
        #     print(">> Use regular base!")
        #     base_text_features = _get_base_text_features(
        #         cfg, classnames, clip_model, text_encoder
        #     )
        # else:
        #     print(">> Use enhanced base!")
        #     base_text_features = _get_enhanced_base_text_features(
        #         cfg,
        #         classnames,
        #         clip_model,
        #         text_encoder,
        #         cfg.TRAINER.TaskRes.ENHANCED_BASE,
        #     )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)

    def forward(self, image):
        device = image.device
        vision_dtype = _vision_dtype_from_module(self.vision)
        try:
            image_features = self.image_encoder(image.to(dtype=vision_dtype))
        except:  # noqa
            image_features = self.image_encoder(image.float())

        # TaskRes-Text
        text_features = self.prompt_learner()

        # # TaskRes-Image
        # text_features, image_res = self.prompt_learner()
        # image_features += image_res

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=image_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=image_features.dtype
            )

        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomQuiltCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.dtype = next(self.image_encoder.parameters()).dtype
        base_text_features = _get_base_text_features_quilt(
            cfg, classnames, clip_model, tokenizer
        )

        # if cfg.TRAINER.TaskRes.ENHANCED_BASE == "none":
        #     print(">> Use regular base!")
        #     base_text_features = _get_base_text_features(
        #         cfg, classnames, clip_model, text_encoder
        #     )
        # else:
        #     print(">> Use enhanced base!")
        #     base_text_features = _get_enhanced_base_text_features(
        #         cfg,
        #         classnames,
        #         clip_model,
        #         text_encoder,
        #         cfg.TRAINER.TaskRes.ENHANCED_BASE,
        #     )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)

    def forward(self, image):
        device = image.device
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:  # noqa
            image_features = self.image_encoder(image.float())

        # TaskRes-Text
        text_features = self.prompt_learner()

        # # TaskRes-Image
        # text_features, image_res = self.prompt_learner()
        # image_features += image_res

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits
        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = conch_model
        self.tokenizer = tokenizer

        base_text_features = _get_base_text_features_conch(
            cfg, classnames, conch_model, tokenizer
        )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)
        self.logit_scale = getattr(conch_model, "logit_scale", None)
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, image):
        device = image.device
        imf = self.clip_model.encode_image(image.to(dtype=self.dtype))
        imf = imf / imf.norm(dim=-1, keepdim=True)

        tf = self.prompt_learner()
        tf = tf / tf.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device, dtype=imf.dtype)
        else:
            logit_scale = self.logit_scale.exp().to(device=device, dtype=imf.dtype)

        return logit_scale * (imf @ tf.t())


class CustomPubMedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model

        self.image_encoder = clip_model.get_image_features
        self.logit_scale = getattr(clip_model, "logit_scale", None)
        self.vision_dtype = next(self.clip_model.vision_model.parameters()).dtype

        base_text_features = _get_base_text_features_pubmedclip(
            cfg, classnames, clip_model, tokenizer
        )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)

    def forward(self, image):
        device = image.device

        try:
            image_features = self.image_encoder(
                pixel_values=image.to(dtype=self.vision_dtype)
            )
        except Exception:
            image_features = self.image_encoder(pixel_values=image.float())

        # TaskRes-Text
        text_features = self.prompt_learner()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=image_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=image_features.dtype
            )

        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class TaskRes(TrainerX):
    """Context Optimization (TaskRes).

    Task Residual for Tuning Vision-Language Models
    https://arxiv.org/abs/2211.10277
    """

    @torch.no_grad()
    def _csv_log_batch(self, batch, logits, y_true):
        probs = torch.softmax(logits.float(), dim=1)
        pmax, y_pred = probs.max(dim=1)
        correct = (y_pred == y_true).long()

        model_name = f"{self.cfg.TRAINER.NAME}_{self.cfg.MODEL.BACKBONE.NAME}"

        B = y_true.size(0)
        for i in range(B):
            path = batch["impath"][i] if "impath" in batch else ""
            pid = infer_patient_id_from_path(path)

            self._csv_rows.append(
                {
                    "model": model_name,
                    "path": str(path),
                    "patient_id": str(pid),
                    "y_true": int(y_true[i].item()),
                    "y_pred": int(y_pred[i].item()),
                    "pmax": float(pmax[i].item()),
                    "correct": int(correct[i].item()),
                }
            )

    def _csv_flush(self, split="test"):
        outdir = Path(self.cfg.OUTPUT_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        fpath = outdir / f"predictions_{split}.csv"

        cols = ["model", "path", "patient_id", "y_true", "y_pred", "pmax", "correct"]
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self._csv_rows)

        print(f"[CSV] wrote {len(self._csv_rows)} rows -> {fpath}")

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TaskRes.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.TaskRes.PREC == "fp32" or cfg.TRAINER.TaskRes.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.MODEL.BACKBONE.NAME == "BiomedCLIP":
            tokenizer = get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.model = CustomBiomedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Quilt-B/32":
            tokenizer = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")
            self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Quilt-B/16":
            tokenizer = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-16")
            self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "PubMedCLIP-B/32":
            tokenizer = CLIPTokenizerFast.from_pretrained(
                "flaviagiammarino/pubmed-clip-vit-base-patch32"
            )
            self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "PLIP-B/32":
            tokenizer = CLIPTokenizerFast.from_pretrained("vinid/plip")
            self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Conch":
            tokenizer = conch.open_clip_custom.get_tokenizer()
            self.model = CustomConchCLIP(cfg, classnames, clip_model, tokenizer)

        else:
            self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()

        self._csv_rows = []

        # Debug sanity check
        batch_debug = next(iter(self.train_loader_x))
        sanity_check_taskres(self, batch=batch_debug, verbose=True)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.TaskRes.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline + CSV logging."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        self._csv_rows = []

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)

            # ---- evaluator (original Dassl) ----
            self.evaluator.process(output, label)

            # ---- CSV logging (ajout minimal) ----
            self._csv_log_batch(batch, output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        # ---- write CSV à la fin ----
        # self._csv_flush(split=split)

        return list(results.values())[0]

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.TaskRes.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(
                'Loading weights to {} from "{}" (epoch = {})'.format(
                    name, model_path, epoch
                )
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
