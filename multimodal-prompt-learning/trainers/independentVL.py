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


def count_params(model):
    """
    Count the total number of parameters and the number of trainable parameters.
    Arg:
        model: model whose parameters are counted.
    Return:
        total: total number of parameters.
        trainable: number of trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_params_by_predicate(model, pred):
    """
    Count the total number of parameters and trainable parameters that satisfy a given predicate.
    Arg:
        model: model whose parameters are counted.
        pred: predicate applied to parameter names and tensors.
    Return:
        total: total number of matching parameters.
        trainable: number of trainable matching parameters.
    """
    tot = 0
    tr = 0
    found = False
    for name, p in model.named_parameters():
        if pred(name, p):
            found = True
            tot += p.numel()
            if p.requires_grad:
                tr += p.numel()
    return (int(tot), int(tr)) if found else None


def print_param_report(model, prefix="[PARAMS]"):
    """
    Print a summary of the total, trainable, and grouped parameters of the model.
    Arg:
        m: model to inspect.
        prefix: prefix used in printed messages.
    Return:
        None
    """
    tot, tr = count_params(model)
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
        c = count_params_by_predicate(model, pred)
        if c is not None:
            kt, ktr = c
            kpct = 100.0 * ktr / max(kt, 1)
            print(f"{prefix} {k:>13s} total/trainable = {kt:,} / {ktr:,} ({kpct:.4f}%)")


def _get_openclip_token_embedding(model):
    """
    Retrieve the token embedding layer from an OpenCLIP-like model.
    Arg:
        model: OpenCLIP-like model.
    Return:
        token_embedding: token embedding layer.
    """
    if hasattr(model, "token_embedding"):
        return model.token_embedding
    if hasattr(model, "text") and hasattr(model.text, "token_embedding"):
        return model.text.token_embedding

    if (
        hasattr(model, "text")
        and hasattr(model.text, "transformer")
        and hasattr(model.text.transformer, "token_embedding")
    ):
        return model.text.transformer.token_embedding

    if hasattr(model, "text") and hasattr(model.text, "transformer"):
        tr = model.text.transformer

        if hasattr(tr, "get_input_embeddings"):
            emb = tr.get_input_embeddings()
            if emb is not None:
                return emb

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


def _get_openclip_text_module(model):
    """
    Retrieve the text module from an OpenCLIP-like model.
    Arg:
        model: OpenCLIP-like model.
    Return:
        text: text module.
    """
    if hasattr(model, "text"):
        return model.text

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


def _get_openclip_positional_embedding(model):
    """
    Retrieve the positional embedding from the text module.
    Arg:
        model: OpenCLIP-like model.
    Return:
        pe: positional embedding tensor.
    """
    text = _get_openclip_text_module(model)
    for name in ["positional_embedding", "pos_embed", "position_embedding"]:
        if hasattr(text, name):
            pe = getattr(text, name)
            if torch.is_tensor(pe):
                return pe
    raise AttributeError("Could not find positional embedding in open_clip text")


def _get_openclip_ln_final(model):
    """
    Retrieve the final layer normalization module from the text tower.
    Arg:
        model: OpenCLIP-like model.
    Return:
        ln_final: final layer normalization module.
    """
    text = _get_openclip_text_module(model)
    for name in ["ln_final", "final_layer_norm", "ln"]:
        if hasattr(text, name):
            return getattr(text, name)
    raise AttributeError("Could not find ln_final in open_clip text")


def _get_openclip_text_projection(model):
    """
    Retrieve the text projection module or tensor from the model.
    Arg:
        model: OpenCLIP-like model.
    Return:
        text_projection: text projection module or tensor.
    """
    if hasattr(model, "text_projection"):
        return model.text_projection
    text = _get_openclip_text_module(model)
    if hasattr(text, "text_projection"):
        return text.text_projection
    if hasattr(text, "proj"):
        return text.proj
    return None


def _get_openclip_text_transformer(model):
    """
    Retrieve the text transformer module from the model.
    Arg:
        model: OpenCLIP-like model.
    Return:
        transformer: text transformer module.
    """
    text = _get_openclip_text_module(model)
    if hasattr(text, "transformer"):
        return text.transformer
    if hasattr(text, "resblocks"):
        return text
    raise AttributeError("Could not find text transformer module in open_clip text")


def _openclip_transformer_batch_first(transformer):
    """
    Check whether the transformer attention blocks use batch-first format.
    Arg:
        transformer: transformer module.
    Return:
        batch_first: whether the transformer uses batch-first format.
    """
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
    """
    Retrieve the text context length for a Hugging Face CLIP-like model.
    Arg:
        clip_model: Hugging Face CLIP-like model.
        tokenizer: optional tokenizer associated with the model.
        default: default context length.
    Return:
        context_length: text context length.
    """
    if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
        ml = int(tokenizer.model_max_length)
        if ml > 0 and ml < 10**6:
            return ml

    cfg = getattr(clip_model, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            mpe = getattr(text_cfg, "max_position_embeddings", None)
            if mpe is not None:
                return int(mpe)

    return int(default)


def tokenize_any(tokenizer, texts, context_length=None, device=None):
    """
    Retrieve the text context length for a Hugging Face CLIP-like model.
    Arg:
        clip_model: Hugging Face CLIP-like model.
        tokenizer: optional tokenizer associated with the model.
        default: default context length.
    Return:
        context_length: text context length.
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        out = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        )
        input_ids = out["input_ids"]

    except TypeError:
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
    """
    Load the selected backbone on CPU and optionally inject visual prompt tuning modules.
    Arg:
        cfg: configuration object.
        mode: training mode.
    Return:
        model: loaded model.
        tokenizer: tokenizer associated with the model, if available.
        preprocess: preprocessing function or transform, if available.
    """
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
        """
        Initialize fixed text features computed from hand-crafted prompts.
        Arg:
            classnames: list of class names.
            clip_model: CLIP model used to encode the prompts.
            prompt_prefix: textual prefix used to build prompts.
        Return:
            None
        """
        super().__init__()
        prompts = [f"{prompt_prefix} {c.replace('_', ' ')}." for c in classnames]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts])

        dev = next(clip_model.parameters()).device
        tokenized = tokenized.to(dev)

        text_dtype = clip_model.token_embedding.weight.dtype

        with torch.no_grad():
            feats = clip_model.encode_text(tokenized)
            feats = feats.to(dtype=text_dtype)

            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", feats, persistent=True)

    def forward(self, device, dtype):
        """
        Return the fixed text features on the requested device and dtype.
        Arg:
            device: target device.
            dtype: target dtype.
        Return:
            text_features: fixed text features.
        """
        return self.text_features.to(device=device, dtype=dtype)


class FixedEmbeddingsBiomed(nn.Module):
    def __init__(self, classnames, biomed_model, tokenizer):
        """
        Initialize fixed text embeddings for BiomedCLIP.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            biomed_model: BiomedCLIP model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
            )

            text_features = biomed_model.encode_text(input_ids).to(dtype=dtype)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)

        self.register_buffer("fixed_text_features", text_features, persistent=False)

    def forward(self):
        """
        Return the precomputed fixed text features.
        Return:
            fixed_text_features: fixed text features.
        """
        return self.fixed_text_features


class FixedEmbeddingsQuilt(nn.Module):
    def __init__(self, classnames, quilt_model, tokenizer):
        """
        Initialize fixed text embeddings for a Quilt/OpenCLIP model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            quilt_model: Quilt/OpenCLIP model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        device = next(quilt_model.parameters()).device
        dtype = next(quilt_model.parameters()).dtype

        prompt_prefix = "a photo of a"
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {c}." for c in classnames]

        tok = tokenizer(prompts)
        if isinstance(tok, dict):
            tok = tok.get("input_ids", tok[list(tok.keys())[0]])
        tok = torch.as_tensor(tok, device=device)

        with torch.no_grad():
            text_features = quilt_model.encode_text(tok).to(dtype=dtype)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("fixed_text_features", text_features, persistent=False)

    def forward(self):
        """
        Return the precomputed fixed text features.
        Return:
            fixed_text_features: fixed text features.
        """
        return self.fixed_text_features


class FixedEmbeddingsConch(nn.Module):
    def __init__(self, classnames, model, tokenizer):
        """
        Initialize fixed text embeddings for the Conch model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            model: Conch model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
        """
        Return the precomputed fixed text features.
        Return:
            fixed_text_features: fixed text features.
        """
        return self.fixed_text_features


class FixedEmbeddingsPubMed(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer):
        """
        Initialize fixed text embeddings for a Hugging Face CLIP-based model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Hugging Face CLIP-based model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
            feats = clip_model.get_text_features(
                input_ids=input_ids, attention_mask=attn
            )
            feats = feats.to(dtype=dtype)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        self.register_buffer("fixed_text_features", feats, persistent=False)

    def forward(self):
        """
        Return the precomputed fixed text features.
        Return:
            fixed_text_features: fixed text features.
        """
        return self.fixed_text_features


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        """
        Initialize the text encoder from the CLIP model.
        Arg:
            clip_model: CLIP model containing the text encoder components.
        Return:
            None
        """
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Encode prompt embeddings into text features.
        Arg:
            prompts: embedded prompt tokens.
            tokenized_prompts: tokenized prompts used to identify the end-of-text token.
        Return:
            x: encoded text features.
        """
        tr_dtype = next(self.transformer.parameters()).dtype
        prompts = prompts.to(dtype=tr_dtype)
        pos = self.positional_embedding.to(dtype=tr_dtype)

        x = prompts + pos
        x = x.permute(1, 0, 2)
        x = x.to(dtype=tr_dtype)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(dtype=tr_dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[
            torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection.to(dtype=tr_dtype)

        return x


def _is_hf_text_tower(model):
    """
    Check whether the text tower follows a Hugging Face-style interface.
    Arg:
        model: model to inspect.
    Return:
        is_hf: whether the text tower is Hugging Face-based.
    """
    return (
        hasattr(model, "text")
        and hasattr(model.text, "transformer")
        and hasattr(model.text.transformer, "get_input_embeddings")
    )


class TextEncoderBiomed(nn.Module):
    def __init__(self, biomed_model, pad_id=0):
        """
        Initialize the text encoder wrapper for a Biomed/OpenCLIP-like model.
        Arg:
            biomed_model: model containing the text tower.
            pad_id: padding token id.
        Return:
            None
        """
        super().__init__()
        self.model = biomed_model
        self.pad_id = int(pad_id)

        self.is_hf = _is_hf_text_tower(biomed_model)

        if not self.is_hf:
            self.token_embedding = _get_openclip_token_embedding(biomed_model)
            self.positional_embedding = _get_openclip_positional_embedding(biomed_model)
            self.ln_final = _get_openclip_ln_final(biomed_model)
            self.transformer = _get_openclip_text_transformer(biomed_model)
            self.text_projection = _get_openclip_text_projection(biomed_model)
            self.batch_first = _openclip_transformer_batch_first(self.transformer)
        else:
            self.transformer = biomed_model.text.transformer
            self.text_projection = _get_openclip_text_projection(
                biomed_model
            )

    def _replace_ctx_tokens(self, x, deep_ctx, n_ctx, batch_first):
        """
        Replace context token embeddings with deep prompt embeddings.
        Arg:
            x: token embeddings.
            deep_ctx: deep prompt embeddings.
            n_ctx: number of context tokens.
            batch_first: whether the tensor uses batch-first format.
        Return:
            x: updated token embeddings.
        """
        if batch_first:
            x = x.clone()
            x[:, 1 : 1 + n_ctx, :] = deep_ctx
            return x
        else:
            x = x.clone()
            x[1 : 1 + n_ctx, :, :] = deep_ctx.permute(1, 0, 2)
            return x

    def forward(self, prompts_emb, tokenized_prompts, deep_prompts=None, n_ctx=None):
        """
        Encode text features from prompt embeddings and optional deep prompts.
        Arg:
            prompts_emb: prompt embeddings.
            tokenized_prompts: tokenized prompts.
            deep_prompts: optional deep prompt embeddings.
            n_ctx: number of context tokens.
        Return:
            feats: encoded text features.
        """
        device = prompts_emb.device
        attn = (tokenized_prompts != self.pad_id).long().to(device=device)

        tr = self.transformer
        x = prompts_emb

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

        ext = attn[:, None, None, :].to(dtype=x.dtype)
        ext = (1.0 - ext) * torch.finfo(x.dtype).min

        hidden = x
        for i, layer in enumerate(layers):
            if use_deep and (1 <= i < depth):
                dc = deep_prompts[i - 1].to(
                    device=hidden.device, dtype=hidden.dtype
                )
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

        last = hidden
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
    def __init__(self, conch_model, pad_id=0):
        """
        Initialize the text encoder wrapper for a Conch/OpenCLIP-like model.
        Arg:
            conch_model: model containing the text tower.
            pad_id: padding token id.
        Return:
            None
        """
        super().__init__()
        self.model = conch_model
        self.pad_id = int(pad_id)

        self.is_hf = _is_hf_text_tower(conch_model)

        if not self.is_hf:
            self.token_embedding = _get_openclip_token_embedding(conch_model)
            self.positional_embedding = _get_openclip_positional_embedding(conch_model)
            self.ln_final = _get_openclip_ln_final(conch_model)
            self.transformer = _get_openclip_text_transformer(conch_model)
            self.text_projection = _get_openclip_text_projection(conch_model)
            self.batch_first = _openclip_transformer_batch_first(self.transformer)
        else:
            self.transformer = conch_model.text.transformer
            self.text_projection = _get_openclip_text_projection(
                conch_model
            )

    def _replace_ctx_tokens(self, x, deep_ctx, n_ctx, batch_first):
        """
        Replace context token embeddings with deep prompt embeddings.
        Arg:
            x: token embeddings.
            deep_ctx: deep prompt embeddings.
            n_ctx: number of context tokens.
            batch_first: whether the tensor uses batch-first format.
        Return:
            x: updated token embeddings.
        """
        if batch_first:
            x = x.clone()
            x[:, 1 : 1 + n_ctx, :] = deep_ctx
            return x
        else:
            x = x.clone()
            x[1 : 1 + n_ctx, :, :] = deep_ctx.permute(1, 0, 2)
            return x

    def forward(self, prompts_emb, tokenized_prompts, deep_prompts=None, n_ctx=None):
        """
        Encode text features from prompt embeddings and optional deep prompts.
        Arg:
            prompts_emb: prompt embeddings.
            tokenized_prompts: tokenized prompts.
            deep_prompts: optional deep prompt embeddings.
            n_ctx: number of context tokens.
        Return:
            feats: encoded text features.
        """
        device = prompts_emb.device
        attn = (tokenized_prompts != self.pad_id).long().to(device=device)

        x = prompts_emb.to(dtype=self.positional_embedding.dtype)

        pos = self.positional_embedding.to(
            device=device, dtype=x.dtype
        )
        if pos.dim() == 2:
            x = x + pos.unsqueeze(0)
        else:
            x = x + pos

        use_deep = (
            (deep_prompts is not None)
            and (n_ctx is not None)
            and (deep_prompts.numel() > 0)
        )
        if use_deep:
            pass

        tr = self.transformer
        blocks = tr.resblocks if hasattr(tr, "resblocks") else tr
        batch_first = bool(
            getattr(getattr(blocks[0], "attn", None), "batch_first", False)
        )

        v_depth = 1
        if use_deep:
            v_depth = min(int(deep_prompts.size(0)) + 1, len(blocks) + 1)

        if not batch_first:
            x = x.permute(1, 0, 2)

        for i, blk in enumerate(blocks):
            if use_deep and (1 <= i < v_depth):
                dc = deep_prompts[i - 1].to(
                    device=device, dtype=x.dtype
                )
                n_ctx_eff = dc.size(1)

                if batch_first:
                    x = x.clone()
                    x[:, 1 : 1 + n_ctx_eff, :] = dc
                else:
                    x = x.clone()
                    x[1 : 1 + n_ctx_eff, :, :] = dc.permute(1, 0, 2)

            x = blk(x)

        if not batch_first:
            x = x.permute(1, 0, 2)

        x = self.ln_final(x)

        idx = (attn.sum(dim=1) - 1).clamp(min=0)
        feats = x[torch.arange(x.size(0), device=device), idx, :]

        proj = self.text_projection
        if proj is not None:
            if torch.is_tensor(proj):
                feats = feats @ proj.to(dtype=feats.dtype, device=device)
            elif isinstance(proj, nn.Module):
                feats = proj(feats)

        return feats


class TextEncoderHFCLIP(nn.Module):
    def __init__(self, clip_model, pad_id=0):
        """
        Encode text features from prompt embeddings and optional deep prompts.
        Arg:
            prompts_emb: prompt embeddings.
            tokenized_prompts: tokenized prompts.
            deep_prompts: optional deep prompt embeddings.
            n_ctx: number of context tokens.
        Return:
            feats: encoded text features.
        """
        super().__init__()
        self.clip = clip_model
        self.pad_id = int(pad_id)

        self.text_model = clip_model.text_model
        self.layers = self.text_model.encoder.layers
        self.text_projection = clip_model.text_projection

    def forward(
        self,
        prompts_emb,
        input_ids,
        attention_mask,
        deep_ctx,
        n_ctx,
    ):
        """
        Encode text features from prompt embeddings and optional deep context prompts.
        Arg:
            prompts_emb: prompt embeddings.
            input_ids: tokenized prompt ids.
            attention_mask: attention mask associated with the prompts.
            deep_ctx: optional deep context prompts.
            n_ctx: number of context tokens.
        Return:
            feats: encoded text features.
        """
        device = prompts_emb.device
        x = prompts_emb

        use_deep = (deep_ctx is not None) and (deep_ctx.numel() > 0) and (n_ctx > 0)
        depth_eff = min((deep_ctx.size(0) + 1) if use_deep else 1, len(self.layers) + 1)

        attn = attention_mask[:, None, None, :].to(dtype=x.dtype, device=device)
        attn = (1.0 - attn) * torch.finfo(x.dtype).min

        hidden = x
        for i, layer in enumerate(self.layers):
            if use_deep and (1 <= i < depth_eff):
                dc = deep_ctx[i - 1].to(
                    device=device, dtype=hidden.dtype
                )
                n_ctx_eff = dc.size(1)
                hidden = hidden.clone()
                hidden[:, 1 : 1 + n_ctx_eff, :] = dc

            sig = inspect.signature(layer.forward)
            kwargs = {}
            if "attention_mask" in sig.parameters:
                kwargs["attention_mask"] = attn
            if "causal_attention_mask" in sig.parameters:
                kwargs["causal_attention_mask"] = None
            out = layer(hidden, **kwargs)
            hidden = out[0] if isinstance(out, (tuple, list)) else out

        eos_pos = input_ids.argmax(dim=-1)
        pooled = hidden[torch.arange(hidden.size(0), device=device), eos_pos]

        feats = self.text_projection(pooled)
        return feats


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        """
        Initialize the vision-language prompt learner for the original CLIP model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: CLIP backbone model.
        Return:
            None
        """
        super().__init__()
        n_cls = len(classnames)
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
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
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
        )
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """
        Initialize the vision-language prompt learner for the original CLIP model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: CLIP backbone model.
        Return:
            None
        """
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
        """
        Build shallow prompt embeddings for all classes.
        Return:
            prompts: prompt embeddings for all classes.
        """
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
        """
        Initialize the vision-language prompt learner for BiomedCLIP.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            biomed_model: BiomedCLIP model.
            hidden_size: text embedding dimension.
            tokenizer: tokenizer associated with the model.
            word_embeddings: text embedding layer.
        Return:
            None
        """
        super().__init__()
        device = next(biomed_model.parameters()).device

        self.cfg = cfg
        self.n_cls = len(classnames)

        self.n_ctx = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        self.depth_t = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT

        dtype = word_embeddings.weight.dtype

        if ctx_init and (self.n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))

            tok = tokenizer([ctx_init])
            if isinstance(tok, dict):
                tok = tok["input_ids"]
            tok = tok.to(device)
            ids = tok[0]
            content = ids[1:]
            ids_ctx = content[:n_ctx]

            with torch.no_grad():
                ctx_vectors = word_embeddings(ids_ctx).to(dtype)
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, hidden_size, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        if self.depth_t > 1:
            deep_ctx = torch.empty(
                self.depth_t - 1, self.n_ctx, hidden_size, dtype=dtype
            )
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)
        else:
            self.ctx_deep = None

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = tokenizer(prompts)
        if isinstance(tokenized_prompts, dict):
            tokenized_prompts = tokenized_prompts["input_ids"]
        tokenized_prompts = tokenized_prompts.to(device)

        self.tokenized_prompts = tokenized_prompts

        with torch.no_grad():
            embedding = word_embeddings(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )

        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """
        Construct full prompt embeddings from prefix, context, and suffix.
        Arg:
            ctx: learnable context embeddings.
            prefix: fixed prefix embeddings.
            suffix: fixed suffix embeddings.
            label: optional class label index.
        Return:
            prompts: complete prompt embeddings.
        """
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
        """
        Build shallow prompt embeddings for all classes.
        Return:
            prompts: prompt embeddings for all classes.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class QuiltVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        """
        Initialize the vision-language prompt learner for a Quilt/OpenCLIP model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            quilt_model: Quilt/OpenCLIP model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        self.n_cls = len(classnames)

        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        n_ctx = int(cfg.TRAINER.IVLP.N_CTX_TEXT)
        depth_t = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)

        self.n_ctx = n_ctx
        self.depth_t = depth_t

        dtype = next(quilt_model.parameters()).dtype
        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = int(token_embedding.weight.shape[1])

        if ctx_init and n_ctx > 0:
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx

            tok = tokenizer([ctx_init])
            if isinstance(tok, dict):
                tok = torch.as_tensor(tok.get("input_ids", tok[list(tok.keys())[0]]))
            else:
                tok = torch.as_tensor(tok)

            with torch.no_grad():
                emb = token_embedding(tok).type(dtype)
            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        if self.depth_t > 1:
            deep_ctx = torch.empty(self.depth_t - 1, self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)
        else:
            self.ctx_deep = None

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]

        tokenized = tokenizer(prompts)
        if isinstance(tokenized, dict):
            tokenized = torch.as_tensor(
                tokenized.get("input_ids", tokenized[list(tokenized.keys())[0]])
            )
        else:
            tokenized = torch.as_tensor(tokenized)
        self.tokenized_prompts = tokenized

        with torch.no_grad():
            embedding = token_embedding(self.tokenized_prompts).type(
                dtype
            )

        self.register_buffer(
            "token_prefix", embedding[:, :1, :], persistent=False
        )
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )

    def construct_prompts(self, ctx_for_classes, prefix, suffix):
        """
        Construct full prompt embeddings from prefix, context, and suffix.
        Arg:
            ctx_for_classes: class-specific context embeddings.
            prefix: fixed prefix embeddings.
            suffix: fixed suffix embeddings.
        Return:
            prompts: complete prompt embeddings.
        """
        return torch.cat([prefix, ctx_for_classes, suffix], dim=1)

    def forward(self):
        """
        Build prompt embeddings and optional deep prompts for all classes.
        Return:
            prompts: prompt embeddings for all classes.
            deep: optional deep prompt embeddings.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class ConchVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        """
        Initialize the vision-language prompt learner for the Conch model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            conch_model: Conch model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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

        te = conch_model.text.token_embedding

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
            )["input_ids"]

            with torch.no_grad():
                emb = te(tok).type(dtype)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        if self.depth_t > 1:
            deep_ctx = torch.empty(self.depth_t - 1, self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)
        else:
            self.ctx_deep = None

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]

        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]

        self.tokenized_prompts = tokenized

        with torch.no_grad():
            embedding = te(self.tokenized_prompts).type(dtype)

        self.register_buffer(
            "token_prefix", embedding[:, :1, :], persistent=False
        )
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx_for_classes, prefix, suffix):
        """
        Construct full prompt embeddings from prefix, context, and suffix.
        Arg:
            ctx_for_classes: class-specific context embeddings.
            prefix: fixed prefix embeddings.
            suffix: fixed suffix embeddings.
        Return:
            prompts: complete prompt embeddings.
        """
        return torch.cat([prefix, ctx_for_classes, suffix], dim=1)

    def forward(self):
        """
        Build prompt embeddings and optional deep prompts for all classes.
        Return:
            prompts: prompt embeddings for all classes.
            deep: optional deep prompt embeddings.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class HFVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        """
        Initialize the vision-language prompt learner for a Hugging Face CLIP-based model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Hugging Face CLIP-based model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
            )["input_ids"]

            with torch.no_grad():
                emb = self.token_embedding(tok)

            ctx_vectors = emb[0, 1 : 1 + self.n_ctx, :].clone()
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(self.n_ctx, self.hidden, dtype=dtype)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        if self.depth_t > 1:
            deep_ctx = torch.empty(
                self.depth_t - 1, self.n_ctx, self.hidden, dtype=dtype
            )
            nn.init.normal_(deep_ctx, std=0.02)
            self.ctx_deep = nn.Parameter(deep_ctx)
        else:
            self.ctx_deep = None

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}.".strip() for name in classnames]

        tok_full = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.tokenized_prompts = tok_full["input_ids"]
        self.attention_mask = tok_full["attention_mask"]

        with torch.no_grad():
            embedding = self.token_embedding(
                self.tokenized_prompts
            )

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :], persistent=False
        )

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx_for_classes, prefix, suffix):
        """
        Construct full prompt embeddings from prefix, context, and suffix.
        Arg:
            ctx_for_classes: class-specific context embeddings.
            prefix: fixed prefix embeddings.
            suffix: fixed suffix embeddings.
        Return:
            prompts: complete prompt embeddings.
        """
        return torch.cat([prefix, ctx_for_classes, suffix], dim=1)

    def forward(self):
        """
        Build prompt embeddings and optional deep prompts for all classes.
        Return:
            prompts: prompt embeddings for all classes.
            deep: optional deep prompt embeddings.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        if self.ctx_deep is None:
            return prompts, None

        deep = self.ctx_deep.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return prompts, deep


class TimmVisionVPT(nn.Module):
    def __init__(
        self, timm_vit, n_ctx, v_depth, return_tokens=False
    ):
        """
        Initialize the visual prompt tuning wrapper for a timm vision transformer.
        Arg:
            timm_vit: timm vision transformer.
            n_ctx: number of visual prompt tokens.
            v_depth: number of layers using visual prompts.
            return_tokens: whether to return all tokens instead of the pooled feature.
        Return:
            None
        """
        super().__init__()
        self.vit = timm_vit
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.return_tokens = return_tokens

        embed_dim = getattr(timm_vit, "embed_dim", None)

        if embed_dim is None:
            embed_dim = getattr(timm_vit, "width", None)

        if embed_dim is None and hasattr(timm_vit, "conv1"):
            embed_dim = timm_vit.conv1.out_channels

        if embed_dim is None and hasattr(timm_vit, "patch_embed"):
            pe = timm_vit.patch_embed
            if hasattr(pe, "proj") and hasattr(pe.proj, "out_channels"):
                embed_dim = pe.proj.out_channels
            elif hasattr(pe, "proj") and hasattr(pe.proj, "weight"):
                embed_dim = pe.proj.weight.shape[0]

        if embed_dim is None:
            for name in ["pos_embed", "positional_embedding", "position_embedding"]:
                if hasattr(timm_vit, name):
                    pe = getattr(timm_vit, name)
                    if torch.is_tensor(pe) and pe.ndim >= 2:
                        embed_dim = pe.shape[-1]
                        break

        if embed_dim is None:
            for _, p in timm_vit.named_parameters():
                if p.ndim == 2 and p.shape[0] >= 64 and p.shape[1] >= 64:
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
        """
        Append the initial visual prompt tokens to the token sequence.
        Arg:
            x: input token sequence.
        Return:
            x: token sequence with appended visual prompts.
        """
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt_for_layer(self, x, layer_idx):
        """
        Replace the visual prompt tokens for a specific transformer layer.
        Arg:
            x: input token sequence.
            layer_idx: transformer layer index.
        Return:
            x: token sequence with updated visual prompts.
        """
        x_prefix = x[:, : x.size(1) - self.n_ctx, :]
        vpt = self.VPT_layers[layer_idx - 1].to(dtype=x.dtype, device=x.device)
        vpt = vpt.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([x_prefix, vpt], dim=1)

    def forward(self, x):
        """
        Encode images with visual prompt tuning.
        Arg:
            x: input image batch.
        Return:
            feat: encoded visual features or tokens.
        """
        x = self.vit.patch_embed(x)

        if x.dim() == 4:
            B = x.shape[0]
            D = self.embed_dim

            if x.shape[1] == D:
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == D:
                x = x.reshape(B, -1, D)
            else:
                raise RuntimeError(
                    f"[TimmVisionVPT] Unexpected 4D patch_embed shape={tuple(x.shape)} (embed_dim={D})"
                )
        elif x.dim() != 3:
            raise RuntimeError(
                f"[TimmVisionVPT] Unexpected patch_embed output dim={x.dim()} shape={tuple(x.shape)}"
            )

        cls = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)

        if getattr(self.vit, "pos_embed", None) is not None:
            pos = self.vit.pos_embed.to(dtype=x.dtype, device=x.device)
            x = x + pos

        if hasattr(self.vit, "pos_drop") and self.vit.pos_drop is not None:
            x = self.vit.pos_drop(x)

        if self.use:
            x = self._append_vpt0(x)

        if not self.use or self.v_depth == 1:
            for blk in self.vit.blocks:
                x = blk(x)
        else:
            for i, blk in enumerate(self.vit.blocks):
                if 1 <= i < self.v_depth:
                    x = self._replace_vpt_for_layer(x, layer_idx=i)
                x = blk(x)

        x = self.vit.norm(x)
        if self.return_tokens:
            return x
        return x[:, 0]


class OpenClipVisionVPT(nn.Module):
    def __init__(self, visual, n_ctx, v_depth=1):
        """
        Initialize the visual prompt tuning wrapper for an OpenCLIP vision tower.
        Arg:
            visual: OpenCLIP vision tower.
            n_ctx: number of visual prompt tokens.
            v_depth: number of layers using visual prompts.
        Return:
            None
        """
        super().__init__()
        self.base = visual
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

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

    def _append_vpt0_BLD(self, x):
        """
        Append the initial visual prompt tokens to the token sequence.
        Arg:
            x: input token sequence.
        Return:
            x: token sequence with appended visual prompts.
        """
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt_LBD(self, x, layer_idx):
        """
        Replace the visual prompt tokens for a specific transformer layer.
        Arg:
            x: input token sequence.
            layer_idx: transformer layer index.
        Return:
            x: token sequence with updated visual prompts.
        """
        prefix = x[: -self.n_ctx, :, :]
        vpt = self.VPT_layers[layer_idx - 1].to(
            dtype=x.dtype, device=x.device
        )
        vpt = vpt.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.cat([prefix, vpt], dim=0)

    def _replace_vpt_for_layer_BLD(self, x, layer_idx):
        """
        Replace the visual prompt tokens for a specific transformer layer.
        Arg:
            x: input token sequence.
            layer_idx: transformer layer index.
        Return:
            x: token sequence with updated visual prompts.
        """
        prefix = x[:, : -self.n_ctx, :]
        vpt = (
            self.VPT_layers[layer_idx - 1]
            .to(dtype=x.dtype, device=x.device)[None, :, :]
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([prefix, vpt], dim=1)

    def forward(self, x):
        """
        Encode images with visual prompt tuning.
        Arg:
            x: input image batch.
        Return:
            feat: encoded visual features.
        """
        x = x.to(
            dtype=self.base.conv1.weight.dtype, device=self.base.conv1.weight.device
        )
        x = self.base.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls = self.base.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)

        pos = self.base.positional_embedding.to(dtype=x.dtype, device=x.device)
        x = x + pos[: x.size(1), :].unsqueeze(0)

        if hasattr(self.base, "patch_dropout") and self.base.patch_dropout is not None:
            x = self.base.patch_dropout(x)

        if self.use:
            x = self._append_vpt0_BLD(x)

        if getattr(self.base, "ln_pre", None) is not None:
            x = self.base.ln_pre(x)

        blocks = self.base.transformer.resblocks
        blk0 = blocks[0]
        batch_first = getattr(blk0.attn, "batch_first", False)

        v_depth = min(self.v_depth, len(blocks)) if self.use else 0

        if batch_first:
            if (not self.use) or (v_depth == 1):
                for blk in blocks:
                    x = blk(x)
            else:
                for i, blk in enumerate(blocks):
                    if 1 <= i < v_depth:
                        x = self._replace_vpt_for_layer_BLD(x, i)
                    x = blk(x)
        else:
            x = x.permute(1, 0, 2)
            if (not self.use) or (v_depth == 1):
                for blk in blocks:
                    x = blk(x)
            else:
                for i, blk in enumerate(blocks):
                    if 1 <= i < v_depth:
                        x = self._replace_vpt_for_layer_LBD(x, i)
                    x = blk(x)
            x = x.permute(1, 0, 2)

        feat = x[:, 0, :]
        if getattr(self.base, "ln_post", None) is not None:
            feat = self.base.ln_post(feat)

        proj = getattr(self.base, "proj", None)
        if proj is not None:
            feat = feat @ proj

        return feat


class OpenAIVisionVPT(nn.Module):
    def __init__(self, visual_vit, n_ctx, v_depth):
        """
        Initialize the visual prompt tuning wrapper for an OpenAI CLIP vision transformer.
        Arg:
            visual_vit: OpenAI CLIP vision transformer.
            n_ctx: number of visual prompt tokens.
            v_depth: number of layers using visual prompts.
        Return:
            None
        """
        super().__init__()
        self.base = visual_vit
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)

        self.input_resolution = getattr(visual_vit, "input_resolution", None)
        self.output_dim = getattr(visual_vit, "output_dim", None)

        width = getattr(visual_vit, "conv1").out_channels
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        if self.use:
            p0 = torch.empty(self.n_ctx, width)
            nn.init.normal_(p0, std=0.02)
            self.VPT0 = nn.Parameter(p0)

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    pi = torch.empty(self.n_ctx, width)
                    nn.init.normal_(pi, std=0.02)
                    self.VPT_layers.append(nn.Parameter(pi))

    def _append_vpt0_NLD(self, x):
        """
        Append the initial visual prompt tokens to the token sequence (NLD format).
        Arg:
            x: input token sequence of shape (batch_size, num_tokens, dim).
        Return:
            x: token sequence with appended visual prompt tokens along the token dimension.
        """
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt_for_layer_LND(self, x, layer_idx):
        """
        Replace the visual prompt tokens for a specific transformer layer (LND format).
        Arg:
            x: input token sequence of shape (num_tokens, batch_size, dim).
            layer_idx: index of the transformer layer.
        Return:
            x: token sequence where the last n_ctx tokens are replaced by layer-specific prompts.
        """
        prefix = x[: x.shape[0] - self.n_ctx, :, :]
        vpt = self.VPT_layers[layer_idx - 1].to(
            dtype=x.dtype, device=x.device
        )
        vpt = vpt.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.cat([prefix, vpt], dim=0)

    def forward(self, x):
        """
        Encode images with visual prompt tuning.
        Arg:
            x: input image batch.
        Return:
            x: encoded visual features.
        """
        x = x.to(
            dtype=self.base.conv1.weight.dtype, device=self.base.conv1.weight.device
        )
        x = self.base.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.base.class_embedding.to(dtype=x.dtype, device=x.device)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )

        x = x + self.base.positional_embedding.to(dtype=x.dtype, device=x.device)

        if self.use:
            x = self._append_vpt0_NLD(x)

        x = self.base.ln_pre(x)

        x = x.permute(1, 0, 2)

        if (not self.use) or (self.v_depth == 1):
            for blk in self.base.transformer.resblocks:
                x = blk(x)
        else:
            for i, blk in enumerate(self.base.transformer.resblocks):
                if 1 <= i < self.v_depth:
                    x = self._replace_vpt_for_layer_LND(x, layer_idx=i)
                x = blk(x)

        x = x.permute(1, 0, 2)

        x = self.base.ln_post(x[:, 0, :])

        if getattr(self.base, "proj", None) is not None:
            x = x @ self.base.proj

        return x


class HFCLIPVisionVPT(nn.Module):
    def __init__(self, vision_model, n_ctx, v_depth):
        """
        Initialize the visual prompt tuning wrapper for a Hugging Face CLIP vision model.
        Arg:
            vision_model: Hugging Face CLIP vision model.
            n_ctx: number of visual prompt tokens.
            v_depth: number of layers using visual prompts.
        Return:
            None
        """
        super().__init__()
        self.base = vision_model
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        hidden = self.base.embeddings.patch_embedding.out_channels
        if self.use:
            p0 = torch.empty(self.n_ctx, hidden)
            nn.init.normal_(p0, std=0.02)
            self.VPT0 = nn.Parameter(p0)

            self.VPT_layers = nn.ParameterList()
            if self.v_depth >= 2:
                for _ in range(1, self.v_depth):
                    pi = torch.empty(self.n_ctx, hidden)
                    nn.init.normal_(pi, std=0.02)
                    self.VPT_layers.append(nn.Parameter(pi))

    def _append_vpt(self, x):
        """
        Append visual prompt tokens to the token sequence.
        Arg:
            x: input token sequence.
        Return:
            token sequence with appended visual prompts.
        """
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)

    def _replace_vpt(self, x, layer_idx):
        """
        Replace the visual prompt tokens for a specific transformer layer.
        Arg:
            x: input token sequence.
            layer_idx: transformer layer index.
        Return:
            token sequence with updated visual prompts.
        """
        prefix = x[:, : x.size(1) - self.n_ctx, :]
        vpt = self.VPT_layers[layer_idx - 1].to(dtype=x.dtype, device=x.device)
        vpt = vpt.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([prefix, vpt], dim=1)

    def forward(self, pixel_values):
        """
        Encode images with visual prompt tuning.
        Arg:
            pixel_values: input image batch.
        Return:
            encoded visual features.
        """
        x = self.base.embeddings(pixel_values)

        if self.use:
            x = self._append_vpt(x)

        x = self.base.pre_layrnorm(x)

        for i, layer in enumerate(self.base.encoder.layers):
            if self.use and (self.v_depth >= 2) and (1 <= i < self.v_depth):
                x = self._replace_vpt(x, layer_idx=i)

            sig = inspect.signature(layer.forward)
            kwargs = {}
            if "attention_mask" in sig.parameters:
                kwargs["attention_mask"] = None
            if "causal_attention_mask" in sig.parameters:
                kwargs["causal_attention_mask"] = None

            out = layer(x, **kwargs)
            x = out[0] if isinstance(out, (tuple, list)) else out

        x = self.base.post_layernorm(x)
        return x[:, 0, :]


class DinoVisionVPT(nn.Module):
    def __init__(self, dino_vit, n_ctx, v_depth):
        """
        Initialize the visual prompt tuning wrapper for a DINO vision transformer.
        Arg:
            dino_vit: DINO vision transformer.
            n_ctx: number of visual prompt tokens.
            v_depth: number of layers using visual prompts.
        Return:
            None
        """
        super().__init__()
        self.vit = dino_vit
        self.n_ctx = int(n_ctx)
        self.v_depth = int(v_depth)
        self.use = (self.n_ctx > 0) and (self.v_depth > 0)

        embed_dim = getattr(self.vit, "embed_dim", None)
        if embed_dim is None:
            embed_dim = self.vit.patch_embed.proj.out_channels

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

        self.num_features = embed_dim

    def _unpack_tokens(self, x):
        """
        Extract token tensors from the DINO block input or output format.
        Arg:
            x: block input or output.
        Return:
            tokens: extracted token tensor.
            packinfo: metadata used to reconstruct the original format.
        """
        if torch.is_tensor(x):
            return x, None

        if hasattr(x, "x") and torch.is_tensor(x.x):
            return x.x, ("x", x)
        if hasattr(x, "tensors") and torch.is_tensor(x.tensors):
            return x.tensors, ("tensors", x)

        raise TypeError(f"Unsupported Dino output type from patch_embed: {type(x)}")

    def _repack_tokens(self, tokens, packinfo):
        """
        Reconstruct the original object format from token tensors.
        Arg:
            tokens: token tensor.
            packinfo: metadata describing the original format.
        Return:
            out: reconstructed object.
        """
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

    def _append_vpt(self, tokens):
        """
        Append visual prompt tokens to the token sequence.
        Arg:
            tokens: input token sequence.
        Return:
            token sequence with appended visual prompts.
        """
        vpt = self.VPT0.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)
        vpt = vpt.expand(tokens.size(0), -1, -1)
        return torch.cat([tokens, vpt], dim=1)

    def _replace_vpt(self, tokens, layer_idx):
        """
        Replace the visual prompt tokens for a specific transformer layer.
        Arg:
            x: input token sequence.
            layer_idx: transformer layer index.
        Return:
            token sequence with updated visual prompts.
        """
        prefix = tokens[:, : tokens.size(1) - self.n_ctx, :]
        vpt = self.VPT_layers[layer_idx - 1].to(
            dtype=tokens.dtype, device=tokens.device
        )
        vpt = vpt.unsqueeze(0).expand(tokens.size(0), -1, -1)
        return torch.cat([prefix, vpt], dim=1)

    def forward(self, x):
        """
        Encode images with visual prompt tuning.
        Arg:
            x: input image batch.
        Return:
            t: encoded visual features.
        """
        out = self.vit.patch_embed(x)
        tokens, packinfo = self._unpack_tokens(out)

        if self.use:
            tokens = self._append_vpt(tokens)

        out = self._repack_tokens(tokens, packinfo)

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

        t, _ = self._unpack_tokens(out)
        t = self.vit.norm(t)

        if t.size(1) >= 1:
            return t.mean(dim=1)
        return t


class VisionOnlyFromVLM(nn.Module):
    def __init__(
        self,
        cfg,
        classnames,
        backbone,
        backend,
        feat_dim,
        force_fp32_head,
        normalize_feats,
    ):
        """
        Initialize a vision-only classifier built from a vision-language backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            backbone: backbone model.
            backend: backend type.
            feat_dim: feature dimension. If None, it is inferred automatically.
            force_fp32_head: whether to use float32 features in the classification head.
            normalize_feats: whether to normalize image features.
        Return:
            None
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.backend = backend
        self.num_classes = len(classnames)
        self.normalize_feats = bool(normalize_feats)
        self.force_fp32_head = bool(force_fp32_head)

        if feat_dim is None:
            feat_dim = self._infer_feat_dim_cpu()
        self.feat_dim = int(feat_dim)

        self.head = nn.Linear(self.feat_dim, self.num_classes)

    @torch.no_grad()
    def _infer_feat_dim_cpu(self):
        """
        Infer the image feature dimension using a dummy forward pass on CPU.
        Return:
            dim: inferred feature dimension.
        """
        params = list(self.backbone.parameters())
        if len(params) > 0:
            orig_device = params[0].device
            orig_dtype = params[0].dtype
        else:
            orig_device = torch.device("cpu")
            orig_dtype = torch.float32

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

        self.backbone.to(device=orig_device, dtype=orig_dtype)
        return dim

    def encode_image_features(self, image):
        """
        Infer the image feature dimension using a dummy forward pass on CPU.
        Return:
            dim: inferred feature dimension.
        """
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

    def forward(self, image, label=None):
        """
        Encode input images into visual features.
        Arg:
            image: input image batch.
        Return:
            feats: encoded image features.
        """
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
        """
        Initialize the IVLP model based on the original CLIP backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: CLIP backbone model.
        Return:
            None
        """
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
        """
        Compute classification logits or training loss from image and text features.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
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
        """
        Initialize the IVLP model based on BiomedCLIP.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            biomed_model: BiomedCLIP model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = biomed_model
        self.logit_scale = getattr(biomed_model, "logit_scale", None)

        self.t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        self.n_ctx_t = int(cfg.TRAINER.IVLP.N_CTX_TEXT)

        self.dtype = next(biomed_model.parameters()).dtype

        if self.t_depth > 0 and self.n_ctx_t > 0:
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
            self.fixed_text = FixedEmbeddingsBiomed(
                classnames, biomed_model, tokenizer
            )

    def encode_image(self, image):
        """
        Encode the input images into visual features.
        Arg:
            image: input image batch.
        Return:
            image_features: encoded image features.
        """
        m = self.clip_model
        dtype = next(m.parameters()).dtype
        if hasattr(m, "encode_image"):
            return m.encode_image(image.to(dtype=dtype))
        if hasattr(m, "visual"):
            return m.visual(image.to(dtype=dtype))
        raise AttributeError("BiomedCLIP model has no encode_image/visual")

    def forward(self, image, label=None):
        """
        Compute classification logits or training loss from image and text features.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
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


class CustomQuiltCLIP(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        """
        Initialize the IVLP model based on a Quilt/OpenCLIP backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            quilt_model: Quilt/OpenCLIP model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
            )
            self.fixed_text = None
        else:
            self.prompt_learner = None
            self.tokenized_prompts = None
            self.text_encoder = None
            self.fixed_text = FixedEmbeddingsQuilt(
                classnames, quilt_model, tokenizer
            )

    def encode_image(self, image):
        """
        Encode the input images into visual features.
        Arg:
            image: input image batch.
        Return:
            image_features: encoded image features.
        """
        m = self.clip_model
        dtype = next(m.parameters()).dtype
        if hasattr(m, "encode_image"):
            return m.encode_image(image.to(dtype=dtype))
        if hasattr(m, "visual"):
            return m.visual(image.to(dtype=dtype))
        raise AttributeError("Quilt model has no encode_image/visual")

    def forward(self, image, label=None):
        """
        Compute classification logits or training loss from image and text features.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
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
        """
        Initialize the IVLP model based on the Conch backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            conch_model: Conch model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
                classnames, conch_model, tokenizer
            )

    def encode_image(self, image):
        """
        Encode the input images into visual features.
        Arg:
            image: input image batch.
        Return:
            image_features: encoded image features.
        """
        m = self.clip_model
        dtype = next(m.parameters()).dtype

        if hasattr(m, "encode_image"):
            return m.encode_image(image.to(dtype=dtype))

        if hasattr(m, "visual"):
            return m.visual(image.to(dtype=dtype))

        raise AttributeError("Conch model has no encode_image")

    def forward(self, image, label=None):
        """
        Compute classification logits or training loss from image and text features.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
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
    def __init__(self, cfg, classnames, clip_model, tokenizer, pad_id=0):
        """
        Initialize the IVLP model based on a Hugging Face CLIP-based model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Hugging Face CLIP-based model.
            tokenizer: tokenizer associated with the model.
            pad_id: padding token id.
        Return:
            None
        """
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip = clip_model
        self.tokenizer = tokenizer

        self.logit_scale = getattr(self.clip, "logit_scale", None)

        t_depth = int(cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT)
        n_ctx_t = int(cfg.TRAINER.IVLP.N_CTX_TEXT)

        if (t_depth > 0) and (n_ctx_t > 0):
            self.prompt_learner = HFVLPromptLearner(
                cfg, classnames, clip_model, tokenizer
            )
            self.text_encoder = TextEncoderHFCLIP(clip_model, pad_id=pad_id)
        else:
            self.prompt_learner = None
            self.text_encoder = None

            self.fixed_text = FixedEmbeddingsPubMed(
                classnames, clip_model, tokenizer
            )

    def _encode_image(self, image):
        """
        Encode the input images into visual features.
        Arg:
            image: input image batch.
        Return:
            image_features: encoded image features.
        """
        dtype = next(self.clip.parameters()).dtype

        if (
            hasattr(self.clip, "vision_model")
            and self.clip.vision_model.__class__.__name__ == "HFCLIPVisionVPT"
        ):
            cls = self.clip.vision_model(image.to(dtype=dtype))
            feats = self.clip.visual_projection(cls)
            return feats

        return self.clip.get_image_features(pixel_values=image.to(dtype=dtype))

    def forward(self, image, label=None):
        """
        Compute classification logits or training loss from image and text features.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
        device = image.device

        image_features = self._encode_image(image)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        n_ctx = int(self.cfg.TRAINER.IVLP.N_CTX_TEXT)

        if self.prompt_learner is None:
            text_features = self.fixed_text().to(
                device=device, dtype=image_features.dtype
            )
        else:
            prompts_emb, deep = (
                self.prompt_learner()
            )
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
    def __init__(self, cfg, classnames, dino_model):
        """
        Initialize the IVLP model based on a DINO backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            dino_model: DINO backbone model.
        Return:
            None
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = dino_model
        self.num_classes = len(classnames)

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            feat_dim = getattr(self.backbone, "embed_dim", None)
        if feat_dim is None:
            feat_dim = 768

        self.head = nn.Linear(feat_dim, self.num_classes)

    def forward(self, image, label=None):
        """
        Compute classification logits or training loss from image and text features.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
        dtype = next(self.backbone.parameters()).dtype
        feats = self.backbone(image.to(dtype=dtype))
        logits = self.head(feats.float())

        if self.training and (label is not None):
            return F.cross_entropy(logits, label)

        return logits


def unwrap_visual(v):
    """
    Unwrap nested visual prompt tuning wrappers and return the base visual module.
    Arg:
        v: visual module or wrapped visual module.
    Return:
        v: unwrapped visual module.
    """
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
        clip_model, tokenizer, _ = load_clip_to_cpu(cfg, mode)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
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

        for _, p in self.model.named_parameters():
            p.requires_grad_(False)

        name_to_update = "prompt_learner"
        for name, p in self.model.named_parameters():
            if name_to_update in name:
                p.requires_grad_(True)
            elif "VPT" in name:
                p.requires_grad_(True)
            elif ("head" in name) and (mode == "vision-only"):
                p.requires_grad_(True)

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
            optim.step()

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
