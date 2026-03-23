import os.path as osp
import os
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from open_clip import get_tokenizer, create_model_from_pretrained
import conch.open_clip_custom
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor


_tokenizer = _Tokenizer()


def count_params(m: nn.Module):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return int(tot), int(tr)


def count_params_by_predicate(m: nn.Module, pred):
    tot = 0
    tr = 0
    found = False
    for name, p in m.named_parameters():
        if pred(name, p):
            found = True
            tot += p.numel()
            if p.requires_grad:
                tr += p.numel()
    return (int(tot), int(tr)) if found else None


def print_param_report(m: nn.Module, prefix="[PARAMS]"):
    tot, tr = count_params(m)
    pct = 100.0 * tr / max(tot, 1)
    print(f"{prefix} total={tot:,} | trainable={tr:,} | trainable%={pct:.4f}%")

    groups = {
        "VPT vision": lambda n, p: n.startswith("clip_model.visual") and ("VPT" in n),
        "VPT text": lambda n, p: (
            n.startswith("clip_model.transformer") and ("VPT" in n)
        ),
        "prompt_learner": lambda n, p: "prompt_learner" in n,
        "head": lambda n, p: (".head" in n) or n.startswith("head"),
    }

    for k, pred in groups.items():
        c = count_params_by_predicate(m, pred)
        if c is not None:
            kt, ktr = c
            kpct = 100.0 * ktr / max(kt, 1)
            print(f"{prefix} {k:>13s} total/trainable = {kt:,} / {ktr:,} ({kpct:.4f}%)")


@torch.no_grad()
def _is_finite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


@torch.no_grad()
def _max_abs(x: torch.Tensor) -> float:
    return float(x.abs().max().item())


def sanity_check_any_backbone(
    trainer=None,
    model: nn.Module | None = None,
    batch: dict | None = None,
    image_size: int = 224,
    n_classes: int = 5,
    device: str | torch.device | None = None,
    check_grads: bool = True,
    do_one_optim_step: bool = False,
    verbose: bool = True,
):
    """
    Sanity check robuste pour tous tes backbones/wrappers.

    Hypothèses minimales:
      - parse batch: dict avec "img" et optionnellement "label"
      - forward du modèle idéalement: forward(image, label=None)
        (mais on gère aussi forward(image) en fallback)
      - en eval (label=None) le modèle doit retourner logits (Tensor) ou un dict/tuple contenant logits.
      - en train (label != None) le modèle peut retourner loss (Tensor) ou logits (Tensor).

    Notes:
      - Si le modèle est totalement frozen (ex: texte fixed + pas de VPT) -> grads = 0 (normal).
      - Si check_grads=True mais aucun paramètre trainable, on ne crash pas, on log juste.
    """
    assert trainer is not None or model is not None, "Donne trainer ou model"
    m = model if model is not None else trainer.model
    m = m.module if hasattr(m, "module") else m

    # -------------------------
    # device
    # -------------------------
    if device is None:
        if trainer is not None:
            device = trainer.device
        else:
            device = next(m.parameters()).device
    device = torch.device(device)

    # -------------------------
    # batch
    # -------------------------
    if batch is None:
        B = 2
        img = torch.randn(B, 3, image_size, image_size, device=device)
        lab = torch.randint(0, n_classes, (B,), device=device)
        batch = {"img": img, "label": lab}
    else:
        img = batch["img"].to(device)
        lab = batch.get("label", None)
        if lab is not None:
            lab = lab.to(device)

    # -------------------------
    # model dtype (best effort)
    # -------------------------
    try:
        model_dtype = next(m.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    if verbose:
        print("\n" + "=" * 110)
        print("[SANITY] model =", m.__class__.__name__)
        print("[SANITY] device =", device, "| model_dtype =", model_dtype)
        print("[SANITY] img =", tuple(img.shape), img.dtype)

    # Helper: choose image dtype based on likely encoder
    def _infer_img_dtype():
        # cherche un conv1 (ou base.conv1 / vit.conv1) pour prendre son dtype (source de vérité)
        for root in [
            "image_encoder",
            "visual",
            "backbone",
            "clip_model",
            "biomed",
            "clip",
            "coca",
        ]:
            if not hasattr(m, root):
                continue
            enc = getattr(m, root)

            # unwrap common wrappers
            for attr in ["base", "vit", "visual"]:
                if hasattr(enc, attr):
                    enc2 = getattr(enc, attr)
                    if hasattr(enc2, "conv1") and hasattr(enc2.conv1, "weight"):
                        return enc2.conv1.weight.dtype

            # direct conv1
            if hasattr(enc, "conv1") and hasattr(enc.conv1, "weight"):
                return enc.conv1.weight.dtype

            # fallback: first real parameter dtype
            if hasattr(enc, "parameters"):
                try:
                    return next(enc.parameters()).dtype
                except StopIteration:
                    pass

        return model_dtype

    img_for_model = img.to(dtype=_infer_img_dtype())

    # -------------------------
    # helper: safe forward
    # -------------------------
    def _safe_forward(
        model_: nn.Module, image_: torch.Tensor, label_: torch.Tensor | None
    ):
        """
        Essayes dans l'ordre:
          1) model(image, label) si possible
          2) model(image) si signature ne prend pas label
        """
        try:
            return model_(image_, label_)
        except TypeError:
            # forward(image) only
            return model_(image_)

    def _extract_logits(out):
        """
        Retourne Tensor logits si possible.
        """
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0:
            if torch.is_tensor(out[0]):
                return out[0]
        if isinstance(out, dict):
            for k in ["logits", "pred", "output"]:
                if k in out and torch.is_tensor(out[k]):
                    return out[k]
        return None

    def _extract_loss(out):
        """
        Retourne Tensor loss si possible.
        """
        if torch.is_tensor(out) and out.dim() == 0:
            return out
        if (
            isinstance(out, (tuple, list))
            and len(out) > 0
            and torch.is_tensor(out[0])
            and out[0].dim() == 0
        ):
            return out[0]
        if (
            isinstance(out, dict)
            and "loss" in out
            and torch.is_tensor(out["loss"])
            and out["loss"].dim() == 0
        ):
            return out["loss"]
        return None

    # -------------------------
    # 1) EVAL: forward logits
    # -------------------------
    m.eval()
    with torch.no_grad():
        out_eval = _safe_forward(m, img_for_model, None)

    logits = _extract_logits(out_eval)

    if verbose:
        print("[SANITY][EVAL] output type =", type(out_eval))
        if logits is None:
            print("[SANITY][EVAL] could not extract logits tensor.")
        else:
            print("[SANITY][EVAL] logits shape =", tuple(logits.shape))
            print(
                "[SANITY][EVAL] logits finite =",
                _is_finite(logits),
                "| max|logit| =",
                _max_abs(logits),
            )
            if logits.dim() >= 2 and logits.size(0) >= 2:
                print("[SANITY][EVAL] logits[0] =", logits[0].detach().cpu())
                print("[SANITY][EVAL] logits[1] =", logits[1].detach().cpu())
                print(
                    "[SANITY][EVAL] diff logits (max abs) =",
                    float((logits[0] - logits[1]).abs().max().item()),
                )

    # -------------------------
    # 2) TRAIN: forward loss (or logits->loss)
    # -------------------------
    m.train()
    loss = None

    if lab is None:
        if verbose:
            print("[SANITY][TRAIN] no labels provided -> skip loss test.")
    else:
        out_train = _safe_forward(m, img_for_model, lab)

        # cas A: le modèle renvoie directement une loss
        loss = _extract_loss(out_train)

        # cas B: le modèle renvoie des logits -> on calcule CE ici
        if loss is None:
            logits_train = _extract_logits(out_train)
            if logits_train is None:
                raise RuntimeError(
                    f"[SANITY][TRAIN] Train forward returned {type(out_train)} "
                    "but I couldn't extract loss or logits."
                )
            loss = torch.nn.functional.cross_entropy(logits_train, lab)

        if verbose:
            print(
                "[SANITY][TRAIN] loss =",
                float(loss.item()),
                "| finite =",
                _is_finite(loss),
            )

    # -------------------------
    # 3) Check logit_scale if present
    # -------------------------
    if hasattr(m, "logit_scale"):
        ls = getattr(m, "logit_scale")
        if torch.is_tensor(ls):
            with torch.no_grad():
                val = float(ls.detach().float().mean().item())
            if verbose:
                print("[SANITY] has logit_scale Tensor | mean =", val)
        else:
            if verbose:
                print("[SANITY] has logit_scale (non-tensor) =", type(ls))

    # -------------------------
    # 4) Gradients check
    # -------------------------
    if check_grads and (lab is not None) and (loss is not None):
        trainable = [p for p in m.parameters() if p.requires_grad]
        if len(trainable) == 0:
            if verbose:
                print(
                    "[SANITY][GRAD] No trainable parameters (all frozen) -> skip grads."
                )
        else:
            for p in m.parameters():
                p.grad = None

            loss.backward()

            grads = []
            wrong = []
            for name, p in m.named_parameters():
                if p.requires_grad:
                    hasg = (p.grad is not None) and torch.isfinite(p.grad).all().item()
                    if hasg:
                        grads.append(name)
                else:
                    if (p.grad is not None) and torch.isfinite(p.grad).all().item():
                        wrong.append(name)

            if verbose:
                print(
                    "[SANITY][GRAD] #params with finite grad (requires_grad=True) =",
                    len(grads),
                )
                print("[SANITY][GRAD] example grads:", grads[:12])
                if len(wrong) > 0:
                    print(
                        "[SANITY][GRAD][WARNING] grads on frozen params! examples:",
                        wrong[:12],
                    )

                focus = [
                    n
                    for n in grads
                    if ("prompt_learner" in n) or ("VPT" in n) or ("head" in n)
                ]
                print("[SANITY][GRAD] grads on prompt_learner/VPT/head =", len(focus))
                print("[SANITY][GRAD] example focus:", focus[:12])

    # -------------------------
    # 5) Optional: one optimizer step
    # -------------------------
    if do_one_optim_step:
        if trainer is None or not hasattr(trainer, "optim") or trainer.optim is None:
            raise ValueError("do_one_optim_step=True requires trainer.optim")
        if loss is None:
            raise ValueError("Need labels/loss for optimizer step")

        trainer.optim.step()
        trainer.optim.zero_grad(set_to_none=True)
        if verbose:
            print("[SANITY][OPTIM] one step OK")

    if verbose:
        print("=" * 110 + "\n")

    return {
        "logits_shape": None if logits is None else tuple(logits.shape),
        "loss": None if loss is None else float(loss.item()),
        "logits_finite": None if logits is None else _is_finite(logits),
        "loss_finite": None if loss is None else _is_finite(loss),
    }


def _get_openclip_token_embedding(model):
    # 1) open_clip classic
    if hasattr(model, "token_embedding"):
        return model.token_embedding
    if hasattr(model, "text") and hasattr(model.text, "token_embedding"):
        return model.text.token_embedding

    # 2) open_clip sometimes nests differently
    if (
        hasattr(model, "text")
        and hasattr(model.text, "transformer")
        and hasattr(model.text.transformer, "token_embedding")
    ):
        return model.text.transformer.token_embedding

    # 3) HF-style text tower (e.g., PubMedBERT inside BiomedCLIP)
    # open_clip HFTextEncoder often exposes `.transformer` as a HF PreTrainedModel
    if hasattr(model, "text") and hasattr(model.text, "transformer"):
        tr = model.text.transformer

        # preferred: HF API
        if hasattr(tr, "get_input_embeddings"):
            emb = tr.get_input_embeddings()
            if emb is not None:
                return emb

        # common attribute paths
        for path in [
            ("embeddings", "word_embeddings"),
            ("text_model", "embeddings", "word_embeddings"),
            ("bert", "embeddings", "word_embeddings"),
            ("roberta", "embeddings", "word_embeddings"),
            ("model", "embeddings", "word_embeddings"),
        ]:
            obj = tr
            ok = True
            for a in path:
                if hasattr(obj, a):
                    obj = getattr(obj, a)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, nn.Embedding):
                return obj

    raise AttributeError(
        "Could not find token_embedding/word_embeddings in this model. "
        "Print dir(model.text) / dir(model.text.transformer) to see actual attribute names."
    )


def _get_openclip_text_module(model: nn.Module) -> nn.Module:
    """
    open_clip peut exposer le texte:
      - soit via model.text (layout récent)
      - soit directement sur model (layout CLIP-style / certains checkpoints hf-hub)
    """
    if hasattr(model, "text"):
        return model.text

    # layout "CLIP-style" : attributs texte au top-level
    has_top_level_text = (
        hasattr(model, "token_embedding")
        and hasattr(model, "transformer")
        and (hasattr(model, "positional_embedding") or hasattr(model, "pos_embed"))
        and (
            hasattr(model, "ln_final")
            or hasattr(model, "final_layer_norm")
            or hasattr(model, "ln")
        )
    )
    if has_top_level_text:
        return model

    raise AttributeError(
        "open_clip model has no .text module and no top-level text attributes "
        "(token_embedding/transformer/positional_embedding/ln_final). "
        "Print(dir(model)) to locate the text tower."
    )


def _get_openclip_positional_embedding(model: nn.Module) -> torch.Tensor:
    text = _get_openclip_text_module(model)
    for name in ["positional_embedding", "pos_embed", "position_embedding"]:
        if hasattr(text, name):
            pe = getattr(text, name)
            if torch.is_tensor(pe):
                return pe
    raise AttributeError("Could not find positional embedding in open_clip text")


def _get_openclip_ln_final(model: nn.Module) -> nn.Module:
    text = _get_openclip_text_module(model)
    for name in ["ln_final", "final_layer_norm", "ln"]:
        if hasattr(text, name):
            return getattr(text, name)
    raise AttributeError("Could not find ln_final in open_clip text")


def _get_openclip_text_projection(model: nn.Module):
    # projection can be on model or model.text depending on checkpoint
    if hasattr(model, "text_projection"):
        return model.text_projection
    text = _get_openclip_text_module(model)
    if hasattr(text, "text_projection"):
        return text.text_projection
    if hasattr(text, "proj"):
        return text.proj
    return None  # sometimes projection is identity / absent


def _get_openclip_text_transformer(model: nn.Module) -> nn.Module:
    text = _get_openclip_text_module(model)
    # open_clip TextTransformer often has .transformer (with resblocks)
    if hasattr(text, "transformer"):
        return text.transformer
    # sometimes it is the transformer itself
    if hasattr(text, "resblocks"):
        return text
    raise AttributeError("Could not find text transformer module in open_clip text")


def _openclip_transformer_batch_first(transformer: nn.Module) -> bool:
    # best-effort: inspect first block's attn
    blocks = None
    if hasattr(transformer, "resblocks"):
        blocks = transformer.resblocks
    elif hasattr(transformer, "blocks"):
        blocks = transformer.blocks
    if blocks is None or len(blocks) == 0:
        return False
    blk0 = blocks[0]
    attn = getattr(blk0, "attn", None)
    return bool(getattr(attn, "batch_first", False)) if attn is not None else False


def _hf_context_length(clip_model, tokenizer=None, default=77):
    # 1) tokenizer (souvent le plus fiable)
    if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
        ml = int(tokenizer.model_max_length)
        if ml > 0 and ml < 10**6:  # HF met parfois 1e30
            return ml

    # 2) config CLIP
    cfg = getattr(clip_model, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            mpe = getattr(text_cfg, "max_position_embeddings", None)
            if mpe is not None:
                return int(mpe)

    # 3) fallback classique CLIP
    return int(default)


def tokenize_any(tokenizer, texts, context_length=None, device=None):
    """
    Retourne input_ids (B, L) torch.LongTensor.
    - Si tokenizer HF: utilise padding/truncation.
    - Si tokenizer open_clip: tokenizer(texts) direct, puis pad/truncate à context_length.
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        # HuggingFace style
        out = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        )
        input_ids = out["input_ids"]

    except TypeError:
        # open_clip style (pas de kwargs)
        input_ids = tokenizer(texts)
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if context_length is not None:
            L = input_ids.shape[1]
            if L < context_length:
                pad = torch.zeros(
                    input_ids.size(0), context_length - L, dtype=input_ids.dtype
                )
                input_ids = torch.cat([input_ids, pad], dim=1)
            elif L > context_length:
                input_ids = input_ids[:, :context_length]

    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids.long()


def load_clip_to_cpu(cfg, mode):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    n_ctx_v = cfg.TRAINER.IVLP.N_CTX_VISION
    n_ctx_t = cfg.TRAINER.IVLP.N_CTX_TEXT
    v_depth = cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION
    t_depth = cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT

    if backbone_name == "Biomedclip":
        model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tokenizer = get_tokenizer(model_id)
        if (v_depth > 0) and (n_ctx_v > 0):
            model.visual.trunk = TimmVisionVPT(
                model.visual.trunk, n_ctx=n_ctx_v, v_depth=v_depth
            )

        return model, tokenizer, preprocess

    elif backbone_name == "Quilt-B/16":
        model_id = "hf-hub:wisdomik/QuiltNet-B-16"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tokenizer = get_tokenizer(model_id)

        if (v_depth > 0) and (n_ctx_v > 0):
            model.visual = OpenClipVisionVPT(
                model.visual, n_ctx=n_ctx_v, v_depth=v_depth
            )

        return model, tokenizer, preprocess

    elif backbone_name == "Quilt-B/32":
        model_id = "hf-hub:wisdomik/QuiltNet-B-32"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tokenizer = get_tokenizer(model_id)

        if (v_depth > 0) and (n_ctx_v > 0):
            model.visual = OpenClipVisionVPT(
                model.visual, n_ctx=n_ctx_v, v_depth=v_depth
            )

        return model, tokenizer, preprocess

    elif backbone_name == "Conch":
        model, preprocess = conch.open_clip_custom.create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=os.environ.get("HF_TOKEN", None),
        )
        tokenizer = conch.open_clip_custom.get_tokenizer()
        if (v_depth > 0) and (n_ctx_v > 0):
            model.visual.trunk = TimmVisionVPT(
                model.visual.trunk, n_ctx=n_ctx_v, v_depth=v_depth, return_tokens=True
            )

        return model, tokenizer, preprocess

    elif backbone_name == "PubMedCLIP-B/32":
        model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        if (v_depth > 0) and (n_ctx_v > 0):
            model.vision_model = HFCLIPVisionVPT(
                model.vision_model, n_ctx=n_ctx_v, v_depth=v_depth
            )

        return model, tokenizer, preprocess

    elif backbone_name == "PLIP-B/32":
        model_id = "vinid/plip"
        model = CLIPModel.from_pretrained(model_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        if (v_depth > 0) and (n_ctx_v > 0):
            model.vision_model = HFCLIPVisionVPT(
                model.vision_model, n_ctx=n_ctx_v, v_depth=v_depth
            )

        return model, tokenizer, preprocess

    elif "DinoBloom" in backbone_name:
        if backbone_name == "DinoBloom-S":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_s.bin"
            embed_dim = 384
        elif backbone_name == "DinoBloom-B":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_b.bin"
            embed_dim = 768
        elif backbone_name == "DinoBloom-L":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_l.bin"
            embed_dim = 1024
        elif backbone_name == "DinoBloom-G":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_g.bin"
            embed_dim = 1536
        else:
            print("Problem")
        ckpt = torch.load(path, map_location="cpu")

        num_tokens = int(1 + (224 / 14) ** 2)
        model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        model.load_state_dict(ckpt, strict=True)

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if (v_depth > 0) and (n_ctx_v > 0) and (mode == "vision-only"):
            model = DinoVisionVPT(model, n_ctx=n_ctx_v, v_depth=v_depth)
        return model, None, preprocess

    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        if (v_depth > 0) and (n_ctx_v > 0) and mode == "vision-only":
            model = clip.build_model(state_dict or model.state_dict())
            model.visual = OpenAIVisionVPT(model.visual, n_ctx=n_ctx_v, v_depth=v_depth)

        else:
            design_details = {
                "trainer": "IVLP",
                "vision_depth": v_depth,
                "language_depth": t_depth,
                "vision_ctx": n_ctx_v,
                "language_ctx": n_ctx_t,
            }
            model = clip.build_model(state_dict or model.state_dict(), design_details)

        return model, None, None


class FixedTextFeatures(nn.Module):
    def __init__(self, classnames, clip_model, prompt_prefix="a photo of a"):
        super().__init__()
        prompts = [f"{prompt_prefix} {c.replace('_', ' ')}." for c in classnames]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, 77)

        dev = next(clip_model.parameters()).device
        tokenized = tokenized.to(dev)

        # dtype "vérité" pour le texte = dtype du transformer texte
        text_dtype = clip_model.token_embedding.weight.dtype

        with torch.no_grad():
            feats = clip_model.encode_text(tokenized)
            feats = feats.to(dtype=text_dtype)

            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", feats, persistent=True)

    def forward(self, device, dtype):
        return self.text_features.to(device=device, dtype=dtype)


class FixedEmbeddingsBiomed(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, tokenizer):
        super().__init__()
        device = next(biomed_model.parameters()).device
        dtype = next(biomed_model.parameters()).dtype

        context_length = getattr(biomed_model, "context_length", None)
        if context_length is None:
            context_length = getattr(
                getattr(biomed_model, "text", None), "context_length", 256
            )

        prompt_prefix = "a photo of a"
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {c}." for c in classnames]

        with torch.no_grad():
            input_ids = tokenize_any(
                tokenizer, prompts, context_length=context_length, device=device
            )  # (n_cls, L)

            text_features = biomed_model.encode_text(input_ids).to(dtype=dtype)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)

        self.register_buffer("fixed_text_features", text_features, persistent=False)

    def forward(self):
        return self.fixed_text_features


class FixedEmbeddingsQuilt(nn.Module):
    """
    Texte fixed (par classe) calculé une fois avec open_clip / Quilt.
    Utilisé quand PROMPT_DEPTH_TEXT == 0 (ou N_CTX_TEXT == 0).
    """

    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        super().__init__()
        device = next(quilt_model.parameters()).device
        dtype = next(quilt_model.parameters()).dtype

        prompt_prefix = "a photo of a"
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {c}." for c in classnames]

        # tokenizer open_clip: selon version, retourne tensor ou dict
        tok = tokenizer(prompts)
        if isinstance(tok, dict):
            tok = tok.get("input_ids", tok[list(tok.keys())[0]])
        tok = torch.as_tensor(tok, device=device)

        with torch.no_grad():
            # open_clip fournit encode_text
            text_features = quilt_model.encode_text(tok).to(dtype=dtype)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("fixed_text_features", text_features, persistent=False)

    def forward(self):
        return self.fixed_text_features


class FixedEmbeddingsConch(nn.Module):
    """
    Texte fixed (par classe) calculé une fois avec open_clip / conch.
    Utilisé quand PROMPT_DEPTH_TEXT == 0.
    """

    def __init__(self, cfg, classnames, model, tokenizer):
        super().__init__()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        max_len = (
            model.context_length
            if hasattr(model, "context_length")
            else model.text.context_length
        )
        prompt_prefix = "a photo of a"
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {c}." for c in classnames]

        tok = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)

        with torch.no_grad():
            text_features = model.encode_text(input_ids).to(dtype=dtype)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("fixed_text_features", text_features, persistent=False)

    def forward(self):
        return self.fixed_text_features


class FixedEmbeddingsPubMed(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        device = next(clip_model.parameters()).device
        dtype = next(clip_model.parameters()).dtype
        max_len = _hf_context_length(clip_model, tokenizer, default=77)
        prompt_prefix = "a photo of a"
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {c}." for c in classnames]

        tok = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)

        with torch.no_grad():
            # HF CLIP: get_text_features renvoie déjà la projection
            feats = clip_model.get_text_features(
                input_ids=input_ids, attention_mask=attn
            )
            feats = feats.to(dtype=dtype)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        self.register_buffer("fixed_text_features", feats, persistent=False)

    def forward(self):
        return self.fixed_text_features


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        tr_dtype = next(self.transformer.parameters()).dtype
        prompts = prompts.to(dtype=tr_dtype)
        pos = self.positional_embedding.to(dtype=tr_dtype)

        x = prompts + pos
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = x.to(dtype=tr_dtype)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(dtype=tr_dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[
            torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection.to(dtype=tr_dtype)

        return x


def _is_hf_text_tower(model) -> bool:
    return (
        hasattr(model, "text")
        and hasattr(model.text, "transformer")
        and hasattr(model.text.transformer, "get_input_embeddings")
    )


class TextEncoderBiomed(nn.Module):
    """
    Encodeur texte robuste:
    - si text tower = open_clip TextTransformer => chemin "CLIP-like" (ton ancien TextEncoderOpenCLIP)
    - si text tower = HF (PubMedBERT) => on passe inputs_embeds à HF transformer
    """

    def __init__(self, biomed_model: nn.Module, pad_id: int = 0):
        super().__init__()
        self.model = biomed_model
        self.pad_id = int(pad_id)

        self.is_hf = _is_hf_text_tower(biomed_model)

        if not self.is_hf:
            # --- open_clip / CLIP-like ---
            self.token_embedding = _get_openclip_token_embedding(biomed_model)
            self.positional_embedding = _get_openclip_positional_embedding(biomed_model)
            self.ln_final = _get_openclip_ln_final(biomed_model)
            self.transformer = _get_openclip_text_transformer(biomed_model)
            self.text_projection = _get_openclip_text_projection(biomed_model)
            self.batch_first = _openclip_transformer_batch_first(self.transformer)
        else:
            # --- HF / BERT-like ---
            self.transformer = biomed_model.text.transformer  # HF model
            self.text_projection = _get_openclip_text_projection(
                biomed_model
            )  # often exists on biomed_model or text

    def _replace_ctx_tokens(self, x, deep_ctx, n_ctx: int, batch_first: bool):
        # x: (B,L,D) if batch_first else (L,B,D)
        # deep_ctx: (B,n_ctx,D)
        if batch_first:
            x = x.clone()
            x[:, 1 : 1 + n_ctx, :] = deep_ctx
            return x
        else:
            x = x.clone()
            x[1 : 1 + n_ctx, :, :] = deep_ctx.permute(1, 0, 2)  # (n_ctx,B,D)
            return x

    def forward(self, prompts_emb, tokenized_prompts, deep_prompts=None, n_ctx=None):
        device = prompts_emb.device
        attn = (tokenized_prompts != self.pad_id).long().to(device=device)  # (B,L)

        # ---- Garde ton code HF actuel (BERT-like) ----
        tr = self.transformer
        x = prompts_emb  # (B,L,H)

        # locate layers (ton code actuel)
        layers = None
        if hasattr(tr, "encoder") and hasattr(tr.encoder, "layer"):
            layers = tr.encoder.layer
        elif hasattr(tr, "encoder") and hasattr(tr.encoder, "layers"):
            layers = tr.encoder.layers
        elif hasattr(tr, "layers") and isinstance(
            tr.layers, (list, tuple, nn.ModuleList)
        ):
            layers = tr.layers
        else:
            raise AttributeError("Cannot locate HF transformer layers")

        use_deep = (
            (deep_prompts is not None)
            and (n_ctx is not None)
            and (deep_prompts.numel() > 0)
        )
        max_depth = len(layers)
        depth = min(int(deep_prompts.size(0)) + 1, max_depth + 1) if use_deep else 1

        ext = attn[:, None, None, :].to(dtype=x.dtype)  # (B,1,1,L)
        ext = (1.0 - ext) * torch.finfo(x.dtype).min

        hidden = x
        for i, layer in enumerate(layers):
            if use_deep and (1 <= i < depth):
                dc = deep_prompts[i - 1].to(
                    device=hidden.device, dtype=hidden.dtype
                )  # (B,n_ctx,H)
                hidden = hidden.clone()
                hidden[:, 1 : 1 + int(n_ctx), :] = dc

            sig = inspect.signature(layer.forward)
            kwargs = {}
            if "attention_mask" in sig.parameters:
                kwargs["attention_mask"] = ext
            if "return_dict" in sig.parameters:
                kwargs["return_dict"] = False

            out = layer(hidden, **kwargs)
            hidden = (
                out[0]
                if isinstance(out, (tuple, list))
                else getattr(out, "last_hidden_state", out)
            )

        last = hidden  # (B,L,H)
        idx = tokenized_prompts.argmax(dim=-1)
        feats = last[torch.arange(last.size(0), device=last.device), idx, :]

        proj = self.text_projection
        if proj is not None:
            if torch.is_tensor(proj):
                feats = feats @ proj.to(dtype=feats.dtype, device=feats.device)
            elif isinstance(proj, nn.Module):
                feats = proj(feats)
        return feats


class TextEncoderConch(nn.Module):
    """
    Encodeur texte robuste:
    - si text tower = open_clip TextTransformer => chemin "CLIP-like" (ton ancien TextEncoderOpenCLIP)
    - si text tower = HF (PubMedBERT) => on passe inputs_embeds à HF transformer
    """

    def __init__(self, biomed_model: nn.Module, pad_id: int = 0):
        super().__init__()
        self.model = biomed_model
        self.pad_id = int(pad_id)

        self.is_hf = _is_hf_text_tower(biomed_model)

        if not self.is_hf:
            # --- open_clip / CLIP-like ---
            self.token_embedding = _get_openclip_token_embedding(biomed_model)
            self.positional_embedding = _get_openclip_positional_embedding(biomed_model)
            self.ln_final = _get_openclip_ln_final(biomed_model)
            self.transformer = _get_openclip_text_transformer(biomed_model)
            self.text_projection = _get_openclip_text_projection(biomed_model)
            self.batch_first = _openclip_transformer_batch_first(self.transformer)
        else:
            # --- HF / BERT-like ---
            self.transformer = biomed_model.text.transformer  # HF model
            self.text_projection = _get_openclip_text_projection(
                biomed_model
            )  # often exists on biomed_model or text

    def _replace_ctx_tokens(self, x, deep_ctx, n_ctx: int, batch_first: bool):
        # x: (B,L,D) if batch_first else (L,B,D)
        # deep_ctx: (B,n_ctx,D)
        if batch_first:
            x = x.clone()
            x[:, 1 : 1 + n_ctx, :] = deep_ctx
            return x
        else:
            x = x.clone()
            x[1 : 1 + n_ctx, :, :] = deep_ctx.permute(1, 0, 2)  # (n_ctx,B,D)
            return x

    def forward(self, prompts_emb, tokenized_prompts, deep_prompts=None, n_ctx=None):
        device = prompts_emb.device
        attn = (tokenized_prompts != self.pad_id).long().to(device=device)  # (B,L)

        # ==========================
        # ---- OPEN_CLIP / CONCH ----
        # ==========================
        # prompts_emb ici = embeddings (B,L,D) déjà construits par PromptLearner (prefix+ctx+suffix)
        x = prompts_emb.to(dtype=self.positional_embedding.dtype)

        # 1) ajouter positional embedding (CLIP-like)
        pos = self.positional_embedding.to(
            device=device, dtype=x.dtype
        )  # (L,D) ou (1,L,D)
        if pos.dim() == 2:
            x = x + pos.unsqueeze(0)  # (B,L,D)
        else:
            x = x + pos  # déjà broadcastable

        use_deep = (
            (deep_prompts is not None)
            and (n_ctx is not None)
            and (deep_prompts.numel() > 0)
        )
        if use_deep:
            # deep_prompts de ton PromptLearner: (depth-1, n_cls, n_ctx, D)
            # on les appliquera juste avant certains blocks comme tu faisais, mais avec la BONNE mise en forme
            pass

        # 2) transformer: gérer batch_first vs LBD
        tr = self.transformer
        # open_clip text transformer: soit un module avec .resblocks, soit directement une liste
        blocks = tr.resblocks if hasattr(tr, "resblocks") else tr
        batch_first = bool(
            getattr(getattr(blocks[0], "attn", None), "batch_first", False)
        )

        # Option: profondeur effective pour deep prompts
        v_depth = 1
        if use_deep:
            v_depth = min(int(deep_prompts.size(0)) + 1, len(blocks) + 1)

        if not batch_first:
            x = x.permute(1, 0, 2)  # (L,B,D)

        for i, blk in enumerate(blocks):
            if use_deep and (1 <= i < v_depth):
                dc = deep_prompts[i - 1].to(
                    device=device, dtype=x.dtype
                )  # (B, n_ctx_eff, D)
                n_ctx_eff = dc.size(1)

                if batch_first:
                    x = x.clone()
                    x[:, 1 : 1 + n_ctx_eff, :] = dc
                else:
                    x = x.clone()
                    x[1 : 1 + n_ctx_eff, :, :] = dc.permute(1, 0, 2)

            x = blk(x)

        if not batch_first:
            x = x.permute(1, 0, 2)  # (B,L,D)

        # 3) ln_final
        x = self.ln_final(x)

        # 4) prendre le dernier non-pad (robuste)
        idx = (attn.sum(dim=1) - 1).clamp(min=0)
        feats = x[torch.arange(x.size(0), device=device), idx, :]  # (B,D)

        # 5) projection
        proj = self.text_projection
        if proj is not None:
            if torch.is_tensor(proj):
                feats = feats @ proj.to(dtype=feats.dtype, device=device)
            elif isinstance(proj, nn.Module):
                feats = proj(feats)

        return feats


class TextEncoderHFCLIP(nn.Module):
    def __init__(self, clip_model: nn.Module, pad_id: int = 0):
        super().__init__()
        self.clip = clip_model
        self.pad_id = int(pad_id)

        self.text_model = clip_model.text_model
        self.layers = self.text_model.encoder.layers
        self.text_projection = clip_model.text_projection  # Linear

    def forward(
        self,
        prompts_emb: torch.Tensor,  # (n_cls, L, H) inputs_embeds
        input_ids: torch.Tensor,  # (n_cls, L) juste pour eos_pos / mask
        attention_mask: torch.Tensor,  # (n_cls, L)
        deep_ctx: torch.Tensor | None,  # (depth-1, n_cls, n_ctx, H) ou None
        n_ctx: int,
    ) -> torch.Tensor:
        device = prompts_emb.device
        x = prompts_emb

        use_deep = (deep_ctx is not None) and (deep_ctx.numel() > 0) and (n_ctx > 0)
        depth_eff = min((deep_ctx.size(0) + 1) if use_deep else 1, len(self.layers) + 1)

        # HF extended attn mask
        # shape attendue: (B, 1, 1, L)
        attn = attention_mask[:, None, None, :].to(dtype=x.dtype, device=device)
        attn = (1.0 - attn) * torch.finfo(x.dtype).min

        hidden = x
        for i, layer in enumerate(self.layers):
            if use_deep and (1 <= i < depth_eff):
                dc = deep_ctx[i - 1].to(
                    device=device, dtype=hidden.dtype
                )  # (B, n_ctx_eff, H)
                n_ctx_eff = dc.size(1)
                hidden = hidden.clone()
                hidden[:, 1 : 1 + n_ctx_eff, :] = dc

            # versions HF: parfois layer(...) veut attention_mask, parfois aussi causal_attention_mask
            sig = inspect.signature(layer.forward)
            kwargs = {}
            if "attention_mask" in sig.parameters:
                kwargs["attention_mask"] = attn
            if "causal_attention_mask" in sig.parameters:
                kwargs["causal_attention_mask"] = None
            out = layer(hidden, **kwargs)
            hidden = out[0] if isinstance(out, (tuple, list)) else out

        # pool : comme HF CLIP (eot)
        eos_pos = input_ids.argmax(dim=-1)
        pooled = hidden[torch.arange(hidden.size(0), device=device), eos_pos]  # (B,H)

        feats = self.text_projection(pooled)  # (B,D)
        return feats


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, (
            "In Independent VL prompting, Language prompt depth should be >=1"
            "\nPlease use VPT trainer if you want to learn only vision "
            "branch  "
        )
        n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(
            f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}"
        )
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        )  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class BiomedVLPromptLearner(nn.Module):
    def __init__(
        self, cfg, classnames, biomed_model, hidden_size, tokenizer, word_embeddings
    ):
        super().__init__()
        device = next(biomed_model.parameters()).device

        self.cfg = cfg
        self.n_cls = len(classnames)

        self.n_ctx = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        self.depth_t = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT

        dtype = word_embeddings.weight.dtype

        # -------------------------
        # 1) Init ctx (CTX_INIT ou random)
        # -------------------------
        if ctx_init and (self.n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))

            tok = tokenizer([ctx_init])  # open_clip tokenizer -> Tensor (1, L)
            if isinstance(tok, dict):
                tok = tok["input_ids"]
            tok = tok.to(device)
            ids = tok[0]
            content = ids[1:]  # skip CLS
            ids_ctx = content[:n_ctx]  # (n_ctx,)

            with torch.no_grad():
                ctx_vectors = word_embeddings(ids_ctx).to(dtype)  # (n_ctx, H)
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, hidden_size, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        # deep ctx (layers 1..depth_t-1), si depth_t > 1
        if self.depth_t > 1:
            deep_ctx = torch.empty(
                self.depth_t - 1, self.n_ctx, hidden_size, dtype=dtype
            )
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)  # (depth_t-1, n_ctx, dim)
        else:
            self.ctx_deep = None

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = tokenizer(prompts)  # open_clip tokenizer
        if isinstance(tokenized_prompts, dict):
            tokenized_prompts = tokenized_prompts["input_ids"]
        tokenized_prompts = tokenized_prompts.to(device)  # (n_cls, L)

        self.tokenized_prompts = tokenized_prompts

        with torch.no_grad():
            embedding = word_embeddings(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        # (depth-1, n_ctx, dim) -> (depth-1, n_cls, n_ctx, dim)
        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class QuiltVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        super().__init__()
        self.n_cls = len(classnames)

        # prends IVLP si tu es dans IVLP, sinon COOP
        # (adapte si tu veux absolument COOP.*)
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        n_ctx = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        depth_t = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)

        self.n_ctx = n_ctx
        self.depth_t = depth_t

        dtype = next(quilt_model.parameters()).dtype
        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = int(token_embedding.weight.shape[1])

        # -------------------------
        # 1) init ctx (ctx_init ou random) -> ctx0 (pour layer0)
        # -------------------------
        if ctx_init and n_ctx > 0:
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            # si ctx_init a plus de mots que n_ctx, tu peux tronquer ou forcer n_ctx = len(...)
            # je fais comme ton code: n_ctx = nb mots
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx

            tok = tokenizer([ctx_init])
            if isinstance(tok, dict):
                tok = torch.as_tensor(tok.get("input_ids", tok[list(tok.keys())[0]]))
            else:
                tok = torch.as_tensor(tok)

            with torch.no_grad():
                emb = token_embedding(tok).type(dtype)  # (1, L, dim)
            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()  # (n_ctx, dim)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        # -------------------------
        # 2) paramètres deep: ctx_layers
        # -------------------------
        self.ctx = nn.Parameter(ctx_vectors)
        # deep ctx (layers 1..depth_t-1), si depth_t > 1
        if self.depth_t > 1:
            deep_ctx = torch.empty(self.depth_t - 1, self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)  # (depth_t-1, n_ctx, dim)
        else:
            self.ctx_deep = None

        # -------------------------
        # 3) construire tokenized_prompts + token_prefix/suffix (CLIP-like)
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]

        tokenized = tokenizer(prompts)
        if isinstance(tokenized, dict):
            tokenized = torch.as_tensor(
                tokenized.get("input_ids", tokenized[list(tokenized.keys())[0]])
            )
        else:
            tokenized = torch.as_tensor(tokenized)
        self.tokenized_prompts = tokenized  # (n_cls, L)

        with torch.no_grad():
            embedding = token_embedding(self.tokenized_prompts).type(
                dtype
            )  # (n_cls, L, dim)

        # même convention que ton code:
        # prefix = SOS (1 token)
        # suffix = CLS/EOS/..., c-à-d tout après 1+n_ctx
        self.register_buffer(
            "token_prefix", embedding[:, :1, :], persistent=False
        )  # (n_cls,1,dim)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )

    def construct_prompts(self, ctx_for_classes, prefix, suffix):
        # ctx_for_classes: (n_cls, n_ctx, dim)
        return torch.cat([prefix, ctx_for_classes, suffix], dim=1)  # (n_cls, L, dim)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        # (depth-1, n_ctx, dim) -> (depth-1, n_cls, n_ctx, dim)
        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class ConchVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_cls = len(classnames)

        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        n_ctx = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        depth_t = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        max_len = (
            conch_model.context_length
            if hasattr(conch_model, "context_length")
            else conch_model.text.context_length
        )
        self.n_ctx = n_ctx
        self.depth_t = depth_t

        dtype = next(conch_model.parameters()).dtype
        ctx_dim = int(conch_model.text.ln_final.weight.shape[0])

        # embedding module
        te = conch_model.text.token_embedding  # nn.Embedding

        # -------------------------
        # 1) init ctx0 (layer 0)
        # -------------------------
        if ctx_init and (n_ctx > 0):
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx

            tok = tokenizer(
                [ctx_init],
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )["input_ids"]  # (1, 77)

            with torch.no_grad():
                emb = te(tok).type(dtype)  # (1, 77, dim)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()  # après BOS
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        # -------------------------
        # 2) paramètres deep: ctx_layers
        # -------------------------
        self.ctx = nn.Parameter(ctx_vectors)
        # deep ctx (layers 1..depth_t-1), si depth_t > 1
        if self.depth_t > 1:
            deep_ctx = torch.empty(self.depth_t - 1, self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)  # (depth_t-1, n_ctx, dim)
        else:
            self.ctx_deep = None

        # -------------------------
        # 3) tokenized_prompts + token_prefix/suffix (CLIP-like)
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]

        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]  # (n_cls, 77)

        self.tokenized_prompts = tokenized  # (n_cls, 77)

        with torch.no_grad():
            embedding = te(self.tokenized_prompts).type(dtype)  # (n_cls, 77, dim)

        self.register_buffer(
            "token_prefix", embedding[:, :1, :], persistent=False
        )  # BOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )

        # si tu en as besoin ailleurs
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx_for_classes, prefix, suffix):
        # ctx_for_classes: (n_cls, n_ctx, dim)
        return torch.cat([prefix, ctx_for_classes, suffix], dim=1)  # (n_cls, L, dim)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        # (depth-1, n_ctx, dim) -> (depth-1, n_cls, n_ctx, dim)
        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class HFVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_cls = len(classnames)

        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        n_ctx = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        depth_t = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        max_len = _hf_context_length(clip_model, tokenizer, default=77)

        dtype = next(clip_model.parameters()).dtype
        self.hidden = int(clip_model.text_model.config.hidden_size)
        self.token_embedding = clip_model.text_model.embeddings.token_embedding

        self.n_ctx = n_ctx
        self.depth_t = depth_t

        # -------------------------
        # 1) init ctx0 (layer 0)
        # -------------------------
        if ctx_init and (n_ctx > 0):
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx

            tok = tokenizer(
                [ctx_init],
                padding=False,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]  # (1, L)

            with torch.no_grad():
                emb = self.token_embedding(tok)  # (1, L, hidden)

            ctx_vectors = emb[0, 1 : 1 + self.n_ctx, :].clone()  # après BOS
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, self.hidden, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        # -------------------------
        # 2) paramètres deep: ctx_layers
        # -------------------------
        self.ctx = nn.Parameter(ctx_vectors)
        # deep ctx (layers 1..depth_t-1), si depth_t > 1
        if self.depth_t > 1:
            deep_ctx = torch.empty(
                self.depth_t - 1, self.n_ctx, self.hidden, dtype=dtype
            )
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)  # (depth_t-1, n_ctx, dim)
        else:
            self.ctx_deep = None

        # -------------------------
        # 3) tokenized_prompts + prefix/suffix (EXACT CoOp)
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]

        tok_full = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.tokenized_prompts = tok_full["input_ids"]  # (n_cls, 77)
        self.attention_mask = tok_full["attention_mask"]  # (n_cls, 77)

        with torch.no_grad():
            embedding = self.token_embedding(
                self.tokenized_prompts
            )  # (n_cls, 77, hidden)

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx_for_classes, prefix, suffix):
        # ctx_for_classes: (n_cls, n_ctx, dim)
        return torch.cat([prefix, ctx_for_classes, suffix], dim=1)  # (n_cls, L, dim)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        # (depth-1, n_ctx, dim) -> (depth-1, n_cls, n_ctx, dim)
        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class TimmVisionVPT(nn.Module):
    def __init__(
        self, timm_vit: nn.Module, n_ctx: int, v_depth: int, return_tokens=False
    ):
        super().__init__()
        self.vit = timm_vit
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.return_tokens = return_tokens

        # --- infer embed_dim robustly (timm / open_clip / conch) ---
        embed_dim = getattr(timm_vit, "embed_dim", None)

        # open_clip VisionTransformer often uses "width"
        if embed_dim is None:
            embed_dim = getattr(timm_vit, "width", None)

        # many ViTs expose conv1 (openai/open_clip style)
        if embed_dim is None and hasattr(timm_vit, "conv1"):
            embed_dim = timm_vit.conv1.out_channels

        # timm style patch_embed.proj
        if embed_dim is None and hasattr(timm_vit, "patch_embed"):
            pe = timm_vit.patch_embed
            if hasattr(pe, "proj") and hasattr(pe.proj, "out_channels"):
                embed_dim = pe.proj.out_channels
            elif hasattr(pe, "proj") and hasattr(pe.proj, "weight"):
                # Conv2d: (out_channels, in_channels, kH, kW)
                embed_dim = pe.proj.weight.shape[0]

        # open_clip / timm positional embedding
        if embed_dim is None:
            for name in ["pos_embed", "positional_embedding", "position_embedding"]:
                if hasattr(timm_vit, name):
                    pe = getattr(timm_vit, name)
                    if torch.is_tensor(pe) and pe.ndim >= 2:
                        embed_dim = pe.shape[-1]
                        break

        # last resort: scan first weight matrix
        if embed_dim is None:
            for _, p in timm_vit.named_parameters():
                if p.ndim == 2 and p.shape[0] >= 64 and p.shape[1] >= 64:
                    # often (out, in) = (3*D, D) or (D, D)
                    embed_dim = min(p.shape[0], p.shape[1])
                    break

        if embed_dim is None:
            raise ValueError(
                f"Cannot infer embed_dim from vit={timm_vit.__class__.__name__}. "
                f"attrs: {', '.join([a for a in ['embed_dim', 'width', 'conv1', 'patch_embed', 'pos_embed', 'positional_embedding'] if hasattr(timm_vit, a)])}"
            )

        self.use = (self.n_ctx > 0) and (self.v_depth > 0)
        self.embed_dim = int(embed_dim)

        if self.use:
            p0 = torch.empty(self.n_ctx, embed_dim)
            nn.init.normal_(p0, std=0.02)
            self.VPT0 = nn.Parameter(p0)

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    pi = torch.empty(self.n_ctx, embed_dim)
                    nn.init.normal_(pi, std=0.02)
                    self.VPT_layers.append(nn.Parameter(pi))

    def _append_vpt0(self, x):
        # x: (B, L, D)
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt_for_layer(self, x, layer_idx: int):
        # remplace les prompts en FIN de séquence
        # layer_idx: 1..v_depth-1
        x_prefix = x[:, : x.size(1) - self.n_ctx, :]
        vpt = self.VPT_layers[layer_idx - 1].to(dtype=x.dtype, device=x.device)
        vpt = vpt.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([x_prefix, vpt], dim=1)

    def forward(self, x: torch.Tensor):
        # 1) patch embed -> (B, N, D)
        x = self.vit.patch_embed(x)

        # Conch/timm peut renvoyer:
        # - (B, D, H, W)  => BCHW
        # - (B, H, W, D)  => BHWD
        if x.dim() == 4:
            B = x.shape[0]
            D = self.embed_dim

            if x.shape[1] == D:
                # (B, D, H, W) -> (B, N, D)
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == D:
                # (B, H, W, D) -> (B, N, D)
                x = x.reshape(B, -1, D)
            else:
                raise RuntimeError(
                    f"[TimmVisionVPT] Unexpected 4D patch_embed shape={tuple(x.shape)} (embed_dim={D})"
                )
        elif x.dim() != 3:
            raise RuntimeError(
                f"[TimmVisionVPT] Unexpected patch_embed output dim={x.dim()} shape={tuple(x.shape)}"
            )

        # 2) cls token
        cls = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)

        # 3) pos embed sur CLS+PATCHES uniquement
        if getattr(self.vit, "pos_embed", None) is not None:
            pos = self.vit.pos_embed.to(dtype=x.dtype, device=x.device)
            x = x + pos

        if hasattr(self.vit, "pos_drop") and self.vit.pos_drop is not None:
            x = self.vit.pos_drop(x)

        # 4) prompts
        if self.use:
            # shallow = v_depth == 1  (juste VPT0)
            # deep = v_depth >= 2 (VPT0 + replace à chaque couche)
            x = self._append_vpt0(x)

        # 5) blocks timm
        if not self.use or self.v_depth == 1:
            # pas de remplacement par couche
            for blk in self.vit.blocks:
                x = blk(x)
        else:
            # deep: remplacer pour layers 1..v_depth-1
            for i, blk in enumerate(self.vit.blocks):
                if 1 <= i < self.v_depth:
                    x = self._replace_vpt_for_layer(x, layer_idx=i)
                x = blk(x)

        # 6) norm + return CLS
        x = self.vit.norm(x)
        if self.return_tokens:
            return x
        return x[:, 0]


class OpenClipVisionVPT(nn.Module):
    def __init__(self, visual, n_ctx: int, v_depth: int = 1):
        super().__init__()
        self.base = visual
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        # embed dim
        if hasattr(self.base, "width"):
            d = self.base.width
        elif hasattr(self.base, "embed_dim"):
            d = self.base.embed_dim
        elif hasattr(self.base, "conv1"):
            d = self.base.conv1.out_channels
        else:
            raise AttributeError("Can't infer embed dim for VPT")

        if self.use:
            self.VPT0 = nn.Parameter(torch.empty(self.n_ctx, d))
            nn.init.normal_(self.VPT0, std=0.02)

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    p = nn.Parameter(torch.empty(self.n_ctx, d))
                    nn.init.normal_(p, std=0.02)
                    self.VPT_layers.append(p)

    def _append_vpt0_BLD(self, x):  # (B,L,D)
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt_LBD(self, x, layer_idx: int):  # (L,B,D)
        # remplace les n_ctx derniers tokens (en fin)
        prefix = x[: -self.n_ctx, :, :]
        vpt = self.VPT_layers[layer_idx - 1].to(
            dtype=x.dtype, device=x.device
        )  # (n_ctx, D)
        vpt = vpt.unsqueeze(1).expand(-1, x.shape[1], -1)  # (n_ctx, B, D)
        return torch.cat([prefix, vpt], dim=0)

    def _replace_vpt_for_layer_BLD(self, x, layer_idx: int):  # x: (B, L, D)
        prefix = x[:, : -self.n_ctx, :]
        vpt = (
            self.VPT_layers[layer_idx - 1]
            .to(dtype=x.dtype, device=x.device)[None, :, :]
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([prefix, vpt], dim=1)

    def forward(self, x):
        # 1) patchify -> (B, N, D)
        x = x.to(
            dtype=self.base.conv1.weight.dtype, device=self.base.conv1.weight.device
        )
        x = self.base.conv1(x)  # (B,D,gh,gw)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # 2) CLS + pos (sur CLS+PATCHES)
        cls = self.base.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)

        pos = self.base.positional_embedding.to(dtype=x.dtype, device=x.device)
        x = x + pos[: x.size(1), :].unsqueeze(0)

        # 3) patch_dropout (si actif) AVANT prompts
        if hasattr(self.base, "patch_dropout") and self.base.patch_dropout is not None:
            x = self.base.patch_dropout(x)

        # 4) prompts après pos (prompts sans pos)
        if self.use:
            x = self._append_vpt0_BLD(x)  # (B, 1+N+nctx, D)

        # 5) ln_pre
        if getattr(self.base, "ln_pre", None) is not None:
            x = self.base.ln_pre(x)

        # 6) transformer (OpenAI-like) : doit être en (L,B,D)
        blocks = self.base.transformer.resblocks
        blk0 = blocks[0]
        batch_first = getattr(blk0.attn, "batch_first", False)  # <- clé du bug

        v_depth = min(self.v_depth, len(blocks)) if self.use else 0

        if batch_first:
            # blocks attendent (B,L,D)
            if (not self.use) or (v_depth == 1):
                for blk in blocks:
                    x = blk(x)
            else:
                for i, blk in enumerate(blocks):
                    if 1 <= i < v_depth:
                        x = self._replace_vpt_for_layer_BLD(x, i)
                    x = blk(x)
        else:
            # blocks attendent (L,B,D)
            x = x.permute(1, 0, 2)  # (L,B,D)
            if (not self.use) or (v_depth == 1):
                for blk in blocks:
                    x = blk(x)
            else:
                for i, blk in enumerate(blocks):
                    if 1 <= i < v_depth:
                        x = self._replace_vpt_for_layer_LBD(x, i)
                    x = blk(x)
            x = x.permute(1, 0, 2)  # (B,L,D)

        # 7) pool CLS + ln_post + proj
        feat = x[:, 0, :]
        if getattr(self.base, "ln_post", None) is not None:
            feat = self.base.ln_post(feat)

        proj = getattr(self.base, "proj", None)
        if proj is not None:
            feat = feat @ proj

        return feat


class OpenAIVisionVPT(nn.Module):
    """
    Wrapper pour CLIP openai-like visual VisionTransformer:
      - v_depth == 0 : no prompts
      - v_depth == 1 : shallow prompts (VPT0) ajoutés une fois
      - v_depth >= 2 : deep prompts (VPT0 + VPT_layers[1..v_depth-1]) remplacés à chaque couche i=1..v_depth-1

    Conformément à ta version OpenAI:
      - pos_embed appliqué à CLS+PATCHES
      - prompts concaténés APRES pos (donc prompts sans pos)
      - prompts placés à la FIN
    """

    def __init__(self, visual_vit: nn.Module, n_ctx: int, v_depth: int):
        super().__init__()
        self.base = visual_vit
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)

        # proxy attrs utiles
        self.input_resolution = getattr(visual_vit, "input_resolution", None)
        self.output_dim = getattr(visual_vit, "output_dim", None)

        width = getattr(visual_vit, "conv1").out_channels  # embed dim
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        if self.use:
            p0 = torch.empty(self.n_ctx, width)
            nn.init.normal_(p0, std=0.02)
            self.VPT0 = nn.Parameter(p0)  # <-- "VPT" => ton freeze le garde

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    pi = torch.empty(self.n_ctx, width)
                    nn.init.normal_(pi, std=0.02)
                    self.VPT_layers.append(nn.Parameter(pi))

    def _append_vpt0_NLD(self, x):
        # x: (B, L, D)
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)  # prompts à la fin

    def _replace_vpt_for_layer_LND(self, x, layer_idx: int):
        # x: (L, B, D) avec prompts en fin => remplacer les derniers n_ctx tokens
        prefix = x[: x.shape[0] - self.n_ctx, :, :]
        vpt = self.VPT_layers[layer_idx - 1].to(
            dtype=x.dtype, device=x.device
        )  # (n_ctx, D)
        vpt = vpt.unsqueeze(1).expand(-1, x.shape[1], -1)  # (n_ctx, B, D)
        return torch.cat([prefix, vpt], dim=0)

    def forward(self, x: torch.Tensor):
        # Reprend la logique VisionTransformer openai-like
        # 1) patchify
        x = x.to(
            dtype=self.base.conv1.weight.dtype, device=self.base.conv1.weight.device
        )
        x = self.base.conv1(x)  # (B, D, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, grid^2)
        x = x.permute(0, 2, 1)  # (B, grid^2, D)

        # 2) CLS token
        x = torch.cat(
            [
                self.base.class_embedding.to(dtype=x.dtype, device=x.device)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # (B, 1+grid^2, D)

        # 3) pos embed (sur CLS+PATCHES)
        x = x + self.base.positional_embedding.to(dtype=x.dtype, device=x.device)

        # 4) prompts après pos (comme ta version normale)
        if self.use:
            x = self._append_vpt0_NLD(x)

        # 5) pre-norm
        x = self.base.ln_pre(x)

        # 6) transformer en LND
        x = x.permute(1, 0, 2)  # (L, B, D)

        # shallow: on ne remplace jamais
        if (not self.use) or (self.v_depth == 1):
            for blk in self.base.transformer.resblocks:
                x = blk(x)
        else:
            # deep: remplacer prompts pour layers 1..v_depth-1 avant d'appliquer blk i
            for i, blk in enumerate(self.base.transformer.resblocks):
                if 1 <= i < self.v_depth:
                    x = self._replace_vpt_for_layer_LND(x, layer_idx=i)
                x = blk(x)

        # 7) back to NLD
        x = x.permute(1, 0, 2)  # (B, L, D)

        # 8) prendre CLS (index 0)
        x = self.base.ln_post(x[:, 0, :])

        if getattr(self.base, "proj", None) is not None:
            x = x @ self.base.proj

        return x


class HFCLIPVisionVPT(nn.Module):
    """
    HF CLIPVisionTransformer wrapper:
    - v_depth == 0: no prompts
    - v_depth == 1: shallow (VPT0 appended once)
    - v_depth >=2: deep (replace last n_ctx_v tokens before layer i=1..v_depth-1)
    Prompts added AFTER position embeddings (prompts have no pos).
    """

    def __init__(self, vision_model: nn.Module, n_ctx: int, v_depth: int):
        super().__init__()
        self.base = vision_model
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        hidden = self.base.embeddings.patch_embedding.out_channels  # 768
        if self.use:
            p0 = torch.empty(self.n_ctx, hidden)
            nn.init.normal_(p0, std=0.02)
            self.VPT0 = nn.Parameter(p0)  # keep "VPT" in name

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    pi = torch.empty(self.n_ctx, hidden)
                    nn.init.normal_(pi, std=0.02)
                    self.VPT_layers.append(nn.Parameter(pi))

    def _append_vpt(self, x):  # x: (B, L, D)
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt(self, x, layer_idx: int):  # x: (B, L, D)
        prefix = x[:, : x.size(1) - self.n_ctx, :]
        vpt = self.VPT_layers[layer_idx - 1].to(dtype=x.dtype, device=x.device)
        vpt = vpt.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([prefix, vpt], dim=1)

    def forward(self, pixel_values: torch.Tensor):
        # embeddings: patch + cls + pos
        x = self.base.embeddings(pixel_values)  # (B, 1+N, D)

        # add prompts after pos
        if self.use:
            x = self._append_vpt(x)

        x = self.base.pre_layrnorm(x)

        # encoder layers (HF CLIPEncoderLayer peut exiger attention_mask & causal_attention_mask)
        for i, layer in enumerate(self.base.encoder.layers):
            # deep replace prompts for layers 1..v_depth-1
            if self.use and (self.v_depth >= 2) and (1 <= i < self.v_depth):
                x = self._replace_vpt(x, layer_idx=i)

            # passe les kwargs uniquement si nécessaires (compat multi-versions)
            sig = inspect.signature(layer.forward)
            kwargs = {}
            if "attention_mask" in sig.parameters:
                kwargs["attention_mask"] = None
            if "causal_attention_mask" in sig.parameters:
                kwargs["causal_attention_mask"] = None

            out = layer(x, **kwargs)
            x = out[0] if isinstance(out, (tuple, list)) else out

        x = self.base.post_layernorm(x)
        return x[:, 0, :]  # CLS


class DinoVisionVPT(nn.Module):
    """
    VPT sur DinoVisionTransformer.
    - v_depth == 0 : pas de prompts
    - v_depth == 1 : shallow (VPT0 ajouté une fois)
    - v_depth >= 2 : deep (VPT0 + remplacement des prompts avant blocks i=1..v_depth-1)

    Hypothèse: après patch_embed, on obtient une séquence (B, N, D).
    Si Dino renvoie un nested tensor, on tente de le "déballer".
    """

    def __init__(self, dino_vit: nn.Module, n_ctx: int, v_depth: int):
        super().__init__()
        self.vit = dino_vit
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        # embed dim
        embed_dim = getattr(self.vit, "embed_dim", None)
        if embed_dim is None:
            # ton print montre patch_embed.proj out_channels = 768
            embed_dim = self.vit.patch_embed.proj.out_channels

        if self.use:
            p0 = torch.empty(self.n_ctx, embed_dim)
            nn.init.normal_(p0, std=0.02)
            self.VPT0 = nn.Parameter(p0)  # "VPT" dans le nom pour ton freeze

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    pi = torch.empty(self.n_ctx, embed_dim)
                    nn.init.normal_(pi, std=0.02)
                    self.VPT_layers.append(nn.Parameter(pi))

        # proxy: certains frameworks attendent que le backbone expose num_features
        self.num_features = embed_dim

    def _unpack_tokens(self, x):
        # cas 1: déjà tensor (B,N,D)
        if torch.is_tensor(x):
            return x, None

        # cas 2: NestedTensor-like (suivant impl)
        if hasattr(x, "x") and torch.is_tensor(x.x):
            return x.x, ("x", x)
        if hasattr(x, "tensors") and torch.is_tensor(x.tensors):
            return x.tensors, ("tensors", x)

        raise TypeError(f"Unsupported Dino output type from patch_embed: {type(x)}")

    def _repack_tokens(self, tokens, packinfo):
        if packinfo is None:
            return tokens
        kind, obj = packinfo
        if kind == "x":
            obj.x = tokens
            return obj
        if kind == "tensors":
            obj.tensors = tokens
            return obj
        return tokens

    def _append_vpt(self, tokens):  # (B,N,D) -> (B,N+n_ctx,D)
        vpt = self.VPT0.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)
        vpt = vpt.expand(tokens.size(0), -1, -1)
        return torch.cat([tokens, vpt], dim=1)

    def _replace_vpt(self, tokens, layer_idx: int):  # remplace les derniers n_ctx
        prefix = tokens[:, : tokens.size(1) - self.n_ctx, :]
        vpt = self.VPT_layers[layer_idx - 1].to(
            dtype=tokens.dtype, device=tokens.device
        )
        vpt = vpt.unsqueeze(0).expand(tokens.size(0), -1, -1)
        return torch.cat([prefix, vpt], dim=1)

    def forward(self, x: torch.Tensor):
        # 1) patch embed
        out = self.vit.patch_embed(x)  # souvent (B,N,D) ou nested
        tokens, packinfo = self._unpack_tokens(out)

        # 2) prompts
        if self.use:
            tokens = self._append_vpt(tokens)

        out = self._repack_tokens(tokens, packinfo)

        # 3) blocks
        if (not self.use) or (self.v_depth == 1):
            for blk in self.vit.blocks:
                out = blk(out)
        else:
            for i, blk in enumerate(self.vit.blocks):
                if 1 <= i < self.v_depth:
                    t, p = self._unpack_tokens(out)
                    t = self._replace_vpt(t, layer_idx=i)
                    out = self._repack_tokens(t, p)
                out = blk(out)

        # 4) norm
        # on normalise sur tokens (et on renvoie CLS-like = premier token si existant,
        # sinon un pool mean)
        t, _ = self._unpack_tokens(out)
        t = self.vit.norm(t)

        # si Dino n'a pas CLS, c'est souvent mean pool
        if t.size(1) >= 1:
            # choix: soit token[0], soit mean pool; à toi de décider.
            # Je mets mean pool (plus standard Dino) :
            return t.mean(dim=1)
        return t


class VisionOnlyFromVLM(nn.Module):
    def __init__(
        self,
        cfg,
        classnames,
        backbone: nn.Module,
        backend: str = "generic",
        feat_dim: int | None = None,
        force_fp32_head: bool = True,
        normalize_feats: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.backend = backend
        self.num_classes = len(classnames)
        self.normalize_feats = bool(normalize_feats)
        self.force_fp32_head = bool(force_fp32_head)

        # infer feature dim once, safely, in init
        if feat_dim is None:
            feat_dim = self._infer_feat_dim_cpu()
        self.feat_dim = int(feat_dim)

        self.head = nn.Linear(self.feat_dim, self.num_classes)

    @torch.no_grad()
    def _infer_feat_dim_cpu(self) -> int:
        """
        Infer feature dim by a tiny forward on CPU in fp32.
        Then restore original device/dtype.
        """
        params = list(self.backbone.parameters())
        if len(params) > 0:
            orig_device = params[0].device
            orig_dtype = params[0].dtype
        else:
            orig_device = torch.device("cpu")
            orig_dtype = torch.float32

        # IMPORTANT: CPU inference in float32
        self.backbone.to(device="cpu", dtype=torch.float32)
        self.backbone.eval()

        x = torch.randn(
            1,
            3,
            self.cfg.INPUT.SIZE[0],
            self.cfg.INPUT.SIZE[0],
            device="cpu",
            dtype=torch.float32,
        )

        f = self.encode_image_features(x)
        dim = int(f.shape[-1])

        # restore
        self.backbone.to(device=orig_device, dtype=orig_dtype)
        return dim

    def encode_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns projected image features (B, D).
        """
        # HF CLIPModel
        if self.backend == "hf_clip":
            clipm = self.backbone
            dtype = (
                next(clipm.parameters()).dtype
                if any(True for _ in clipm.parameters())
                else image.dtype
            )

            if (
                hasattr(clipm, "vision_model")
                and clipm.vision_model.__class__.__name__ == "HFCLIPVisionVPT"
            ):
                cls = clipm.vision_model(image.to(dtype=dtype))
                feats = clipm.visual_projection(cls)
            else:
                feats = clipm.get_image_features(pixel_values=image.to(dtype=dtype))
            return feats

        # generic CLIP/open_clip/conch/biomed
        if hasattr(self.backbone, "encode_image"):
            p = next(self.backbone.parameters(), None)
            dtype = p.dtype if p is not None else image.dtype
            device = p.device if p is not None else image.device
            return self.backbone.encode_image(image.to(device=device, dtype=dtype))

        if hasattr(self.backbone, "visual"):
            p = next(self.backbone.parameters(), None)
            dtype = p.dtype if p is not None else image.dtype
            device = p.device if p is not None else image.device
            return self.backbone.visual(image.to(device=device, dtype=dtype))

        raise AttributeError("No encode_image/visual found for this backbone")

    def forward(self, image: torch.Tensor, label: torch.Tensor | None = None):
        feats = self.encode_image_features(image)

        if self.normalize_feats:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        feats = self.encode_image_features(image)
        logits = self.head(feats.float() if self.force_fp32_head else feats)

        if self.training and label is not None:
            return F.cross_entropy(logits, label)
        return logits


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = next(clip_model.parameters()).dtype

        self.t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)

        if self.t_depth > 0:
            self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.text_encoder = TextEncoder(clip_model)
        else:
            self.prompt_learner = None
            self.tokenized_prompts = None
            self.text_encoder = None

        self.fixed_text = FixedTextFeatures(classnames, clip_model)

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        if hasattr(self.image_encoder, "base") and hasattr(
            self.image_encoder.base, "conv1"
        ):
            dtype = self.image_encoder.base.conv1.weight.dtype
        elif hasattr(self.image_encoder, "conv1"):
            dtype = self.image_encoder.conv1.weight.dtype
        else:
            dtype = next(self.image_encoder.parameters()).dtype

        image_features = self.image_encoder(image.to(dtype=dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.t_depth == 0:
            text_features = self.fixed_text(
                device=image.device, dtype=image_features.dtype
            )
        else:
            prompts = self.prompt_learner()
            tokenized = self.tokenized_prompts.to(image.device)
            text_features = self.text_encoder(prompts, tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)

        return logits


class CustomBiomedCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = biomed_model  # keep same naming style
        self.logit_scale = getattr(biomed_model, "logit_scale", None)

        self.t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        self.n_ctx_t = int(cfg.TRAINER.IVLP.N_CTX_TEXT)

        # dtype "vérité" (biomed/open_clip souvent fp16)
        self.dtype = next(biomed_model.parameters()).dtype

        # --- text parts ---
        if self.t_depth > 0 and self.n_ctx_t > 0:
            # prompt learner ctx appris
            token_emb = _get_openclip_token_embedding(biomed_model)
            hidden = token_emb.weight.shape[1]
            self.prompt_learner = BiomedVLPromptLearner(
                cfg,
                classnames,
                biomed_model,
                hidden_size=hidden,
                tokenizer=tokenizer,
                word_embeddings=token_emb,
            )
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            pad_id = getattr(tokenizer, "pad_id", 0)
            self.text_encoder = TextEncoderBiomed(biomed_model, pad_id=pad_id)
            self.fixed_text = None
        else:
            self.prompt_learner = None
            self.tokenized_prompts = None
            self.text_encoder = None
            # texte fixe par classe (pré-calc)
            self.fixed_text = FixedEmbeddingsBiomed(
                cfg, classnames, biomed_model, tokenizer
            )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        m = self.clip_model
        dtype = next(m.parameters()).dtype
        if hasattr(m, "encode_image"):
            return m.encode_image(image.to(dtype=dtype))
        if hasattr(m, "visual"):
            return m.visual(image.to(dtype=dtype))
        raise AttributeError("BiomedCLIP model has no encode_image/visual")

    def forward(self, image, label=None):
        # logit_scale
        if self.logit_scale is None:
            # fallback
            logit_scale = torch.tensor(1.0, device=image.device)
        else:
            logit_scale = (
                self.logit_scale.exp()
                if torch.is_tensor(self.logit_scale)
                else torch.exp(self.logit_scale)
            )

        # image feats
        img_feats = self.encode_image(image)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # text feats
        if self.prompt_learner is None:
            txt_feats = self.fixed_text()
        else:
            prompts_emb, deep_prompts = self.prompt_learner()
            tok = self.tokenized_prompts.to(image.device)
            txt_feats = self.text_encoder(
                prompts_emb,
                tok,
                deep_prompts=deep_prompts,
                n_ctx=self.n_ctx_t,  # ou self.prompt_learner.n_ctx
            )
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        logits = logit_scale * (img_feats @ txt_feats.t())

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)
        return logits


class CustomQuiltCLIP(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = quilt_model
        self.tokenizer = tokenizer

        self.t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        self.n_ctx_t = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        self.logit_scale = getattr(quilt_model, "logit_scale", None)

        if self.t_depth > 0 and self.n_ctx_t > 0:
            self.prompt_learner = QuiltVLPromptLearner(
                cfg, classnames, quilt_model, tokenizer
            )
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            pad_id = getattr(tokenizer, "pad_id", 0)
            self.text_encoder = TextEncoderConch(
                quilt_model, pad_id=pad_id
            )  # OK aussi pour open_clip
            self.fixed_text = None
        else:
            self.prompt_learner = None
            self.tokenized_prompts = None
            self.text_encoder = None
            self.fixed_text = FixedEmbeddingsQuilt(
                cfg, classnames, quilt_model, tokenizer
            )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        m = self.clip_model
        dtype = next(m.parameters()).dtype
        if hasattr(m, "encode_image"):
            return m.encode_image(image.to(dtype=dtype))
        if hasattr(m, "visual"):
            return m.visual(image.to(dtype=dtype))
        raise AttributeError("Quilt model has no encode_image/visual")

    def forward(self, image, label=None):
        if self.logit_scale is None:
            logit_scale = torch.tensor(1.0, device=image.device)
        else:
            logit_scale = (
                self.logit_scale.exp()
                if torch.is_tensor(self.logit_scale)
                else torch.exp(self.logit_scale)
            )

        img_feats = self.encode_image(image)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.prompt_learner is None:
            txt_feats = self.fixed_text()
        else:
            prompts_emb, deep_prompts = self.prompt_learner()
            tok = self.tokenized_prompts.to(image.device)
            txt_feats = self.text_encoder(
                prompts_emb,
                tok,
                deep_prompts=deep_prompts,
                n_ctx=self.n_ctx_t,
            )
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        logits = logit_scale * (img_feats @ txt_feats.t())

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)
        return logits


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = conch_model
        self.tokenizer = tokenizer

        self.t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        self.n_ctx_t = int(cfg.TRAINER.IVLP.N_CTX_TEXT)

        self.logit_scale = getattr(conch_model, "logit_scale", None)

        if self.t_depth > 0 and self.n_ctx_t > 0:
            self.prompt_learner = ConchVLPromptLearner(
                cfg, classnames, conch_model, tokenizer
            )

            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            pad_id = getattr(tokenizer, "pad_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "pad_token_id", 0)
            pad_id = int(pad_id)
            self.text_encoder = TextEncoderConch(conch_model, pad_id=pad_id)
            self.fixed_text = None

        else:
            self.prompt_learner = None
            self.tokenized_prompts = None
            self.text_encoder = None

            self.fixed_text = FixedEmbeddingsConch(
                cfg, classnames, conch_model, tokenizer
            )

    def encode_image(self, image):
        m = self.clip_model
        dtype = next(m.parameters()).dtype

        if hasattr(m, "encode_image"):
            return m.encode_image(image.to(dtype=dtype))

        if hasattr(m, "visual"):
            return m.visual(image.to(dtype=dtype))

        raise AttributeError("Conch model has no encode_image")

    def forward(self, image, label=None):

        if self.logit_scale is None:
            logit_scale = torch.tensor(1.0, device=image.device)
        else:
            logit_scale = (
                self.logit_scale.exp()
                if torch.is_tensor(self.logit_scale)
                else torch.exp(self.logit_scale)
            )

        img_feats = self.encode_image(image)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.prompt_learner is None:
            txt_feats = self.fixed_text()

        else:
            prompts_emb, deep_prompts = self.prompt_learner()

            tok = self.tokenized_prompts.to(image.device)

            txt_feats = self.text_encoder(
                prompts_emb,
                tok,
                deep_prompts=deep_prompts,
                n_ctx=self.n_ctx_t,
            )

            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        logits = logit_scale * (img_feats @ txt_feats.t())

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)

        return logits


class CustomPubMedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip = clip_model
        self.tokenizer = tokenizer

        self.logit_scale = getattr(self.clip, "logit_scale", None)

        t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        n_ctx_t = int(cfg.TRAINER.IVLP.N_CTX_TEXT)

        # text encoder (HF) utilisant inputs_embeds
        if (t_depth > 0) and (n_ctx_t > 0):
            self.prompt_learner = HFVLPromptLearner(
                cfg, classnames, clip_model, tokenizer
            )
            self.text_encoder = TextEncoderHFCLIP(clip_model, pad_id=pad_id)
        else:
            self.prompt_learner = None
            self.text_encoder = None

            self.fixed_text = FixedEmbeddingsPubMed(
                cfg, classnames, clip_model, tokenizer
            )

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        dtype = next(self.clip.parameters()).dtype

        # si tu as wrappé clip.vision_model par HFCLIPVisionVPT
        if (
            hasattr(self.clip, "vision_model")
            and self.clip.vision_model.__class__.__name__ == "HFCLIPVisionVPT"
        ):
            cls = self.clip.vision_model(image.to(dtype=dtype))
            feats = self.clip.visual_projection(cls)
            return feats

        return self.clip.get_image_features(pixel_values=image.to(dtype=dtype))

    def forward(self, image, label=None):
        device = image.device

        # image feats
        image_features = self._encode_image(image)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        # text feats
        n_ctx = int(self.cfg.TRAINER.IVLP.N_CTX_TEXT)

        if self.prompt_learner is None:
            text_features = self.fixed_text().to(
                device=device, dtype=image_features.dtype
            )
        else:
            prompts_emb, deep = (
                self.prompt_learner()
            )  # prompts_emb: (n_cls,L,H), deep: (depth-1,n_cls,n_ctx,H) or None
            input_ids = self.prompt_learner.tokenized_prompts.to(device)
            attn = self.prompt_learner.attention_mask.to(device)

            text_features = self.text_encoder(
                prompts_emb.to(device=device),
                input_ids=input_ids,
                attention_mask=attn,
                deep_ctx=deep.to(device=device) if deep is not None else None,
                n_ctx=n_ctx,
            )
            text_features = text_features.to(dtype=image_features.dtype)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)

        # logit scale
        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=image_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=image_features.dtype
            )

        logits = logit_scale * (image_features @ text_features.t())

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)
        return logits


class CustomDinoBloom(nn.Module):
    """
    DinoBloom vision-only:
    - backbone = DinoBloom (optionnellement wrappé avec DinoVisionVPT)
    - head = Linear(D -> n_cls)
    """

    def __init__(self, cfg, classnames, dino_model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = dino_model
        self.num_classes = len(classnames)

        # infer feat dim
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            feat_dim = getattr(self.backbone, "embed_dim", None)
        if feat_dim is None:
            # fallback: typiquement vitb14 -> 768
            feat_dim = 768

        self.head = nn.Linear(feat_dim, self.num_classes)

    def forward(self, image, label=None):
        dtype = next(self.backbone.parameters()).dtype
        feats = self.backbone(image.to(dtype=dtype))  # (B, D)
        logits = self.head(feats.float())  # head en fp32 ok

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)

        return logits


def unwrap_visual(v):
    # déplie tant que c'est ton wrapper
    while isinstance(v, OpenClipVisionVPT):
        v = v.base
    return v


@TRAINER_REGISTRY.register()
class IVLP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        mode = cfg.TRAINER.IVLP.MODE

        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, tokenizer, preprocess = load_clip_to_cpu(cfg, mode)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.MODEL.BACKBONE.NAME == "Biomedclip":
            if mode == "vision-only":
                self.model = VisionOnlyFromVLM(
                    cfg, classnames, clip_model, backend="generic"
                )
            else:
                self.model = CustomBiomedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME in ["Quilt-B/16", "Quilt-B/32"]:
            if mode == "vision-only":
                self.model = VisionOnlyFromVLM(
                    cfg, classnames, clip_model, backend="generic"
                )
            else:
                self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Conch":
            if mode == "vision-only":
                self.model = VisionOnlyFromVLM(
                    cfg, classnames, clip_model, backend="generic"
                )
            else:
                self.model = CustomConchCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME in ["PLIP-B/32", "PubMedCLIP-B/32"]:
            if mode == "vision-only":
                self.model = VisionOnlyFromVLM(
                    cfg, classnames, clip_model, backend="hf_clip"
                )
            else:
                self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif "DinoBloom" in cfg.MODEL.BACKBONE.NAME:
            self.model = CustomDinoBloom(cfg, classnames, clip_model)

        else:
            if mode == "vision-only":
                self.model = VisionOnlyFromVLM(
                    cfg, classnames, clip_model, backend="generic"
                )
            else:
                self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for n, p in self.model.named_parameters():
            if "VPT" in n:
                print(n, p.requires_grad)

        # 1) freeze all
        for _, p in self.model.named_parameters():
            p.requires_grad_(False)

        # 2) unfreeze prompt_learner + all VPT (text+vision) + head if vision-only
        name_to_update = "prompt_learner"
        for name, p in self.model.named_parameters():
            if name_to_update in name:
                p.requires_grad_(True)
            elif "VPT" in name:
                p.requires_grad_(True)
            elif ("head" in name) and (mode == "vision-only"):
                p.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Count of parameters
        m = self.model.module if hasattr(self.model, "module") else self.model
        print_param_report(m, prefix="[PARAMS]")

        # Debug - sanity check
        # sanity_check_any_backbone(
        #     trainer=self,
        #     image_size=cfg.INPUT.SIZE[0],
        #     n_classes=len(self.dm.dataset.classnames),
        # )

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        if self.epoch == 0 and self.batch_idx == 0:
            print(
                "labels:",
                label.detach().cpu().tolist(),
                "unique:",
                label.unique().cpu().tolist(),
            )

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()  # <-- UN SEUL step

        loss_summary = {"loss": loss.item()}

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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(
                'Loading weights to {} from "{}" (epoch = {})'.format(
                    name, model_path, epoch
                )
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
