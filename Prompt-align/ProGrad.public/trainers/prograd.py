import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import CLIPModel, CLIPTokenizerFast
from open_clip import get_tokenizer, create_model_from_pretrained
from torch.nn.modules.loss import _Loss
import conch.open_clip_custom


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


@torch.no_grad()
def _flatten_grads(params):
    """Concatène grad de params en un vecteur 1D (ignore None)."""
    import torch

    vec = []
    for p in params:
        if p.grad is None:
            continue
        vec.append(p.grad.detach().reshape(-1))
    if len(vec) == 0:
        return None
    return torch.cat(vec, dim=0)


def _zero_grads(module):
    for p in module.parameters():
        p.grad = None


def _dot(a, b):
    return float((a * b).sum().item())


def sanity_check_prograd_all_backbones(
    trainer,
    image_size=224,
    batch_size=2,
    verbose=True,
    atol_cos=1e-3,
    atol_dot=1e-3,
    require_projection_effect=True,
):
    """
    Sanity check ProGrad robuste multi-backbone + vérif de projection de gradient.

    Vérifie:
    - teacher frozen, teacher grads = None
    - teacher.text_features bien construites (shape, finite, norm)
    - student: seuls prompt_learner.* trainables
    - forward teacher/student OK (logits BxC finite)
    - KL dépend du teacher (g_kl non nul)
    - projection: g_proj ⟂ g_kl (dot ~ 0 / cos ~ 0)
    - projection change effectivement le gradient (optionnel)
    """
    import torch
    from torch.cuda.amp import autocast

    cfg = trainer.cfg
    device = trainer.device
    backbone = cfg.MODEL.BACKBONE.NAME
    classnames = trainer.dm.dataset.classnames
    n_cls = len(classnames)

    student = trainer.model
    teacher = trainer.zs_clip

    # si DataParallel
    stu = student.module if hasattr(student, "module") else student
    tea = teacher.module if hasattr(teacher, "module") else teacher

    prec = cfg.TRAINER.COOP.PREC
    lam = float(cfg.LOSS.LAMBDA)

    if verbose:
        print("\n" + "=" * 100)
        print("🔍 SANITY CHECK ProGrad (multi-backbone + projection-aware)")
        print("=" * 100)
        print("Backbone:", backbone)
        print("Device:", device, "| n_cls:", n_cls, "| prec:", prec, "| lambda:", lam)

    # -------------------------------------------------
    # 1) Teacher MUST be frozen
    # -------------------------------------------------
    for n, p in tea.named_parameters():
        assert not p.requires_grad, (
            f"[SANITY-ProGrad] Teacher param requires_grad=True: {n}"
        )

    # -------------------------------------------------
    # 2) Build teacher text features
    # -------------------------------------------------
    if getattr(tea, "text_features", None) is None:
        if verbose:
            print("[SANITY-ProGrad] teacher.text_features is None -> building...")
        tea.build_text_features(device)

    tf = tea.text_features
    assert tf is not None, "[SANITY-ProGrad] teacher.text_features is still None"
    assert tf.dim() == 2 and tf.size(0) == n_cls, (
        f"[SANITY-ProGrad] teacher.text_features bad shape: {tuple(tf.shape)}"
    )
    assert torch.isfinite(tf).all(), (
        "[SANITY-ProGrad] teacher.text_features contains NaN/Inf"
    )

    # check normalized (approx)
    tf_norm = tf.norm(dim=-1)
    if verbose:
        print(
            "[SANITY-ProGrad] teacher.text_features:",
            tuple(tf.shape),
            tf.dtype,
            tf.device,
            "| mean(norm)=",
            float(tf_norm.mean().item()),
        )
    assert torch.isfinite(tf_norm).all()
    # pas forcément exactement 1 si float16, on tolère
    assert float((tf_norm - 1.0).abs().mean().item()) < 1e-1, (
        "[SANITY-ProGrad] teacher.text_features not normalized-ish"
    )

    # -------------------------------------------------
    # 3) Student trainables only prompt_learner.*
    # -------------------------------------------------
    trainable = [(n, p) for n, p in stu.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "[SANITY-ProGrad] No trainable params in student"
    if verbose:
        print("[SANITY-ProGrad] Trainable params:")
        for n, p in trainable:
            print(f"  ✔ {n:70s} {tuple(p.shape)} {p.dtype} {p.device}")

    for n, _ in trainable:
        assert "prompt_learner" in n, (
            f"[SANITY-ProGrad] Unexpected trainable param: {n}"
        )

    # liste params prompts (pour grads)
    prompt_params = [p for (n, p) in trainable if "prompt_learner" in n]
    assert len(prompt_params) > 0, "[SANITY-ProGrad] prompt_params empty"

    # -------------------------------------------------
    # 4) Forward check teacher/student
    # -------------------------------------------------
    stu.eval()
    tea.eval()

    x = torch.randn(
        batch_size, 3, image_size, image_size, device=device, dtype=torch.float32
    )
    y = torch.randint(0, n_cls, (batch_size,), device=device)

    with torch.no_grad():
        stu_logits = stu(x)
        tea_logits = tea(x)

    assert torch.is_tensor(stu_logits) and stu_logits.shape == (batch_size, n_cls)
    assert torch.is_tensor(tea_logits) and tea_logits.shape == (batch_size, n_cls)
    assert torch.isfinite(stu_logits).all(), "[SANITY-ProGrad] student logits NaN/Inf"
    assert torch.isfinite(tea_logits).all(), "[SANITY-ProGrad] teacher logits NaN/Inf"

    if verbose:
        print(
            "[SANITY-ProGrad] forward OK | student logits:",
            tuple(stu_logits.shape),
            stu_logits.dtype,
            "| teacher logits:",
            tuple(tea_logits.shape),
            tea_logits.dtype,
        )

    # -------------------------------------------------
    # 5) Gradient decomposition: g_ce and g_kl (sur prompt_learner)
    # -------------------------------------------------
    stu.train()
    tea.eval()

    assert hasattr(trainer, "criterion") and trainer.criterion is not None, (
        "[SANITY-ProGrad] trainer.criterion missing (call after setting it)"
    )

    # compute CE and KL separately to inspect gradients
    _zero_grads(stu)
    _zero_grads(tea)

    if prec == "amp":
        scaler = getattr(trainer, "scaler", None)
        assert scaler is not None, "[SANITY-ProGrad] AMP on but trainer.scaler is None"

        with autocast():
            stu_logits = stu(x)
            with torch.no_grad():
                tea_logits = tea(x)
            xe_loss, kl_loss = trainer.criterion(stu_logits, tea_logits, y)

        # grads for XE
        _zero_grads(stu)
        scaler.scale(xe_loss).backward(retain_graph=True)
        g_xe = _flatten_grads(prompt_params)
        assert g_xe is not None, "[SANITY-ProGrad] g_xe is None"

        # grads for KL
        _zero_grads(stu)
        scaler.scale(kl_loss).backward()
        g_kl = _flatten_grads(prompt_params)
        assert g_kl is not None, "[SANITY-ProGrad] g_kl is None"

    else:
        stu_logits = stu(x)
        with torch.no_grad():
            tea_logits = tea(x)

        xe_loss, kl_loss = trainer.criterion(stu_logits, tea_logits, y)

        # g_xe
        _zero_grads(stu)
        xe_loss.backward(retain_graph=True)
        g_xe = _flatten_grads(prompt_params)
        assert g_xe is not None, "[SANITY-ProGrad] g_xe is None"

        # g_kl
        _zero_grads(stu)
        kl_loss.backward()
        g_kl = _flatten_grads(prompt_params)
        assert g_kl is not None, "[SANITY-ProGrad] g_kl is None"

    assert torch.isfinite(g_xe).all(), "[SANITY-ProGrad] g_xe has NaN/Inf"
    assert torch.isfinite(g_kl).all(), "[SANITY-ProGrad] g_kl has NaN/Inf"

    nx = float(g_xe.norm().item())
    nk = float(g_kl.norm().item())
    if verbose:
        print(
            f"[SANITY-ProGrad] xe_loss={float(xe_loss.item()):.6f} | kl_loss={float(kl_loss.item()):.6f}"
        )
        print(f"[SANITY-ProGrad] ||g_xe||={nx:.6e} | ||g_kl||={nk:.6e}")

    assert nx > 0, "[SANITY-ProGrad] g_xe is zero"

    eps_kl = 1e-6  # ou 1e-7 si tu veux être plus strict

    if nk < eps_kl:
        if verbose:
            print(
                f"[SANITY-ProGrad][WARN] ||g_kl|| is tiny ({nk:.3e}) -> "
                "skip projection orthogonality check (numerically unstable at init)."
            )
        do_projection_checks = False
    else:
        do_projection_checks = True

    # -------------------------------------------------
    # 6) Projection check: g_proj = g_xe - proj_{g_kl}(g_xe)
    #    => g_proj ⟂ g_kl
    # -------------------------------------------------
    if do_projection_checks:
        # --- do projection in float64 for numerical stability ---
        gx = g_xe.double()
        gk = g_kl.double()

        nk64 = gk.norm().clamp_min(1e-12)  # stable
        u = gk / nk64                       # unit vector along g_kl

        # stable projection: g_proj = g_xe - <g_xe,u> u
        coef = (gx * u).sum()
        g_proj = gx - coef * u

        # diagnostics
        dot_after = float((g_proj * gk).sum().item())
        cos_after = float(dot_after / (g_proj.norm().item() * nk64.item() + 1e-12))

        if verbose:
            dot_before = float((gx * gk).sum().item())
            print(f"[SANITY-ProGrad] dot(g_xe, g_kl)={dot_before:.6e}")
            print(f"[SANITY-ProGrad] dot(g_proj, g_kl)={dot_after:.6e} | cos={cos_after:.6e}")

        # --- robust tolerance: relative + absolute floor ---
        rhs_rel = atol_dot * (g_proj.norm().item() * nk64.item() + 1e-12)
        rhs = max(rhs_rel, 1e-10)  # absolute floor to avoid over-strict when nk is small

        assert abs(dot_after) < rhs, (
            f"[SANITY-ProGrad] Projection failed: |dot_after|={abs(dot_after):.3e} >= rhs={rhs:.3e} "
            f"(nk={nk64.item():.3e})"
        )
        assert abs(cos_after) < atol_cos, (
            f"[SANITY-ProGrad] Projection failed: cosine={cos_after:.3e} not near 0"
        )

        effect = float((g_proj - gx).norm().item() / (gx.norm().item() + 1e-12))
        if verbose:
            print(f"[SANITY-ProGrad] projection effect (||g_proj-g_xe||/||g_xe||) = {effect:.6e}")

        if require_projection_effect:
            assert effect > 1e-6, "[SANITY-ProGrad] Projection has no effect (unexpected)."

    # -------------------------------------------------
    # 7) Ensure teacher still has no grads, and no grads outside prompt_learner
    # -------------------------------------------------
    # teacher grads
    for p in tea.parameters():
        assert p.grad is None, (
            "[SANITY-ProGrad] Teacher received gradients (should be None)"
        )

    # grads outside prompt_learner in student
    bad = []
    for name, p in stu.named_parameters():
        if p.grad is not None and ("prompt_learner" not in name):
            bad.append(name)
    assert len(bad) == 0, (
        f"[SANITY-ProGrad] Unexpected grads outside prompt_learner: {bad[:10]}"
    )

    if verbose:
        print(
            "✅ SANITY-ProGrad OK (teacher frozen + multi-backbone forward + gradient projection verified)"
        )
        print("=" * 100 + "\n")

    return True


def is_openclip(backbone_name: str):
    return backbone_name in ["Quilt-B/32", "Quilt-B/16", "BiomedCLIP", "Conch"]


def is_hf_clip(backbone_name: str):
    return backbone_name in ["PubMedCLIP-B/32", "PLIP-B/32"]


def _as_input_ids(tokens):

    if isinstance(tokens, dict):
        tokens = tokens.get("input_ids", list(tokens.values())[0])
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    return torch.as_tensor(tokens, dtype=torch.long)


def _vision_dtype_from_module(vision: torch.nn.Module) -> torch.dtype:
    # OpenAI CLIP / ViT-like : conv1 existe
    if hasattr(vision, "conv1") and hasattr(vision.conv1, "weight"):
        return vision.conv1.weight.dtype
    # fallback robuste
    return next(vision.parameters()).dtype


def _get_openclip_token_embedding(model):
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
    raise AttributeError("Could not find token_embedding in open_clip model")


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


def get_openclip_tokenizer_for_conch(clip_model):
    # 1) si conch expose get_tokenizer()
    if hasattr(conch.open_clip_custom, "get_tokenizer"):
        try:
            return conch.open_clip_custom.get_tokenizer()
        except TypeError:
            pass

    # 2) si conch expose tokenize(...)
    if hasattr(conch.open_clip_custom, "tokenize"):
        fn = conch.open_clip_custom.tokenize

        def _wrapped(texts):
            try:
                return fn(texts)
            except TypeError:
                pass
            try:
                return fn(clip_model, texts)
            except TypeError:
                pass
            tok_obj = getattr(clip_model, "tokenizer", None)
            if tok_obj is not None:
                try:
                    return fn(tok_obj, texts)
                except TypeError:
                    pass
            raise TypeError(f"Unsupported Conch tokenize signature: {fn}")

        return _wrapped

    # 3) fallback open_clip standard
    return get_tokenizer("ViT-B-16")


def _get_input_ids(tokenized, pad_id=0, max_len=77):
    """
    Returns torch.LongTensor (B, L) padded/truncated to max_len.
    Handles:
      - Tensor
      - dict / BatchEncoding with 'input_ids'
      - list[int]
      - list[list[int]] with ragged lengths (Conch)
    """
    # Tensor already
    if torch.is_tensor(tokenized):
        ids = tokenized
        # optionally pad/truncate if 1D or wrong length
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.size(1) != max_len:
            # pad/truncate
            if ids.size(1) > max_len:
                ids = ids[:, :max_len]
            else:
                pad = ids.new_full((ids.size(0), max_len - ids.size(1)), int(pad_id))
                ids = torch.cat([ids, pad], dim=1)
        return ids.long()

    # HF BatchEncoding-like
    if hasattr(tokenized, "data") and isinstance(tokenized.data, dict):
        tokenized = tokenized.data

    # dict-like
    if isinstance(tokenized, dict):
        ids = tokenized.get("input_ids", None)
        if ids is None:
            ids = tokenized[next(iter(tokenized.keys()))]
        return _get_input_ids(ids, pad_id=pad_id, max_len=max_len)

    # Conch tokenize may return python lists
    if isinstance(tokenized, list):
        if len(tokenized) == 0:
            raise ValueError("Empty tokenized list")

        # list[int] -> make batch
        if isinstance(tokenized[0], int):
            row = tokenized[:max_len]
            if len(row) < max_len:
                row = row + [int(pad_id)] * (max_len - len(row))
            return torch.tensor([row], dtype=torch.long)

        # list[list[int]] -> ragged batch -> pad/truncate
        if isinstance(tokenized[0], (list, tuple)):
            padded = []
            for row in tokenized:
                row = list(row)[:max_len]
                if len(row) < max_len:
                    row = row + [int(pad_id)] * (max_len - len(row))
                padded.append(row)
            return torch.tensor(padded, dtype=torch.long)

    # BatchEncoding attribute
    if hasattr(tokenized, "input_ids"):
        return _get_input_ids(tokenized.input_ids, pad_id=pad_id, max_len=max_len)

    raise TypeError(f"Cannot extract input_ids from {type(tokenized)}")


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

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
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            assert n_ctx >= prompt_n_ctx, (
                f"#tokens ({n_ctx}) should larger equal than #initial prompt tokens ({prompt_n_ctx}, {ctx_init})"
            )

            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors[n_ctx - prompt_n_ctx :, :] = embedding[
                0, 1 : 1 + prompt_n_ctx, :
            ]
            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
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
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class BiomedPromptLearner(nn.Module):
    def __init__(
        self, cfg, classnames, biomed_model, hidden_size, tokenizer, word_embeddings
    ):
        super().__init__()
        device = next(biomed_model.parameters()).device

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX  # ou COOP.N_CTX si tu veux
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = word_embeddings.weight.dtype
        self.csc = cfg.TRAINER.COOP.CSC

        # -------------------------
        # 1) Init ctx (CTX_INIT ou random)
        # -------------------------
        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            assert n_ctx >= prompt_n_ctx, (
                f"#tokens ({n_ctx}) should larger equal than #initial prompt tokens ({prompt_n_ctx}, {ctx_init})"
            )

            tok = tokenizer([ctx_init])  # open_clip tokenizer -> Tensor (1, L)
            if isinstance(tok, dict):
                tok = tok["input_ids"]
            tok = tok.to(device)
            ids = tok[0]
            content = ids[1:]  # skip CLS
            ids_ctx = content[:n_ctx]  # (n_ctx,)

            with torch.no_grad():
                ctx_vectors_init = word_embeddings(ids_ctx).to(dtype)  # (n_ctx, H)

            ctx_vectors = torch.zeros(n_ctx, hidden_size, dtype=dtype, device=device)
            ctx_vectors[n_ctx - prompt_n_ctx :, :] = ctx_vectors_init

            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            if self.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, hidden_size, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, hidden_size, dtype=dtype)

            prompt_prefix = " ".join(["X"] * n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

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
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, batch_size, device, dtype):
        ctx = self.ctx.to(device=device, dtype=dtype)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(batch_size, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prefix = self.token_prefix.to(device=device, dtype=dtype)
        suffix = self.token_suffix.to(device=device, dtype=dtype)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class QuiltPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        self.csc = cfg.TRAINER.COOP.CSC

        dtype = next(quilt_model.parameters()).dtype
        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = token_embedding.weight.shape[1]

        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            assert n_ctx >= prompt_n_ctx, (
                f"#tokens ({n_ctx}) should larger equal than #initial prompt tokens ({prompt_n_ctx}, {ctx_init})"
            )

            # --- tokenize (open_clip => (1,77)) ---
            tok = tokenizer([ctx_init])
            if isinstance(tok, dict):
                tok = tok.get("input_ids", list(tok.values())[0])
            if hasattr(tok, "input_ids"):
                tok = tok.input_ids
            tok = torch.as_tensor(tok, dtype=torch.long)  # (1,77) ou (77,)

            ids = tok[0] if tok.ndim == 2 else tok  # (77,)
            ids = ids[1:]  # drop SOS/CLS
            ids = ids[ids != 0]  # drop PAD (souvent 0)

            # drop EOT: on enlève le dernier token restant (heuristique robuste open_clip)
            if ids.numel() > 0:
                ids = ids[:-1]

            # ne garder que prompt_n_ctx
            ids_ctx = ids[:prompt_n_ctx]  # (prompt_n_ctx,)

            with torch.no_grad():
                ctx_vectors_init = token_embedding(ids_ctx).to(
                    dtype=dtype
                )  # (prompt_n_ctx, dim)

            # ctx (n_ctx, dim) puis on colle l'init à la fin
            ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)
            ctx_vectors[n_ctx - prompt_n_ctx : n_ctx, :] = ctx_vectors_init

            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            prompt_prefix = " ".join(["X"] * n_ctx)

            if self.csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        # -------------------------
        # 2) Construire prompts par classe
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized = tokenizer(prompts)
        if isinstance(tokenized, dict):
            tokenized = torch.as_tensor(
                tokenized.get("input_ids", tokenized[list(tokenized.keys())[0]])
            )
        else:
            tokenized = torch.as_tensor(tokenized)

        self.tokenized_prompts = tokenized  # (n_cls, L)
        with torch.no_grad():
            embedding = token_embedding(self.tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class ConchPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        self.tokenizer = tokenizer
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

        # dim texte Conch = 768 (d'après ton print)
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        dtype = next(conch_model.parameters()).dtype

        # -------------------------
        # 1) init ctx
        # -------------------------
        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            tok = tokenizer(
                [ctx_init],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )["input_ids"]  # (1, 77)

            with torch.no_grad():
                ctx_vector_init = conch_model.text.token_embedding(tok).type(
                    dtype
                )  # (1, 77, dim)

            ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors[n_ctx - prompt_n_ctx :, :] = ctx_vector_init[
                0, 1 : 1 + prompt_n_ctx, :
            ]
            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            n_ctx = n_ctx
            prompt_prefix = " ".join(["X"] * n_ctx)

            if self.csc:
                print("Initializing class-specific contexts (Conch)")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context (Conch)")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        # -------------------------
        # 2) tokenized prompts par classe
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"]  # (n_cls, 77)

        self.tokenized_prompts = tokenized

        with torch.no_grad():
            embedding = conch_model.text.token_embedding(self.tokenized_prompts).type(
                dtype
            )

        self.register_buffer("token_prefix", embedding[:, :1, :])  # BOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )  # le reste

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts


class HFPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

        self.token_embedding = clip_model.text_model.embeddings.token_embedding

        # dims
        self.hidden = clip_model.text_model.config.hidden_size  # ex: 512
        self.vis_dim = clip_model.config.projection_dim  # ex: 512

        dtype = self.token_embedding.weight.dtype
        device = self.token_embedding.weight.device

        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            tok = tokenizer(
                [ctx_init],
                padding=False,
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(device)  # (1, L)

            with torch.no_grad():
                ctx_vector_init = self.token_embedding(tok)  # (1, L, hidden)

            ctx_vectors = torch.zeros(n_ctx, self.hidden, device=device, dtype=dtype)

            ctx_vectors[n_ctx - prompt_n_ctx :, :] = ctx_vector_init[
                0, 1 : 1 + prompt_n_ctx, :
            ]
            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"

        else:
            prompt_prefix = " ".join(["X"] * n_ctx)
            if self.csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, self.hidden)
            else:
                ctx_vectors = torch.empty(n_ctx, self.hidden)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        print(f'HF Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # -------- 2) construire prompts string (EXACT CoOp) --------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tok_full = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        self.tokenized_prompts = tok_full["input_ids"].to(device)  # (n_cls, 77)
        self.attention_mask = tok_full["attention_mask"].to(device)  # (n_cls, 77)

        with torch.no_grad():
            embedding = self.token_embedding(self.tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        return torch.cat([prefix, ctx, suffix], dim=1)


class CLIP_ZS(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits


class BiomedCLIP_ZS(nn.Module):
    def __init__(self, cfg, classnames, tokenizer):
        super().__init__()
        print("Loading BiomedCLIP (open_clip)")

        biomed_model = load_clip_to_cpu(cfg)
        biomed_model.float()  # comme toi

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")

        tokens = tokenizer(prompts)
        if isinstance(tokens, dict):
            tokens = tokens.get("input_ids", list(tokens.values())[0])
        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids
        tokens = torch.as_tensor(tokens, dtype=torch.long)

        # calcule text_features sur CPU (puis on mettra sur GPU au forward)
        with torch.no_grad():
            tf = biomed_model.encode_text(tokens)  # (C, D) float
            tf = tf / tf.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", tf, persistent=False)

        self.clip_model = biomed_model  # même nom que ton code
        self.tokenizer = tokenizer

    def forward(self, image):
        device = image.device
        dt = next(self.clip_model.parameters()).dtype

        image_features = self.clip_model.encode_image(image.to(dtype=dt))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = getattr(self.clip_model, "logit_scale", None)
        if logit_scale is None:
            ls = torch.tensor(1 / 0.07, device=device, dtype=image_features.dtype)
        else:
            ls = logit_scale.exp().to(device=device, dtype=image_features.dtype)

        text_features = self.text_features.to(device=device, dtype=image_features.dtype)
        logits = ls * (image_features @ text_features.t())
        return logits


class QuiltCLIP_ZS(nn.Module):
    def __init__(self, cfg, classnames, tokenizer):
        super().__init__()

        print(f"Loading Quilt (open_clip): {cfg.MODEL.BACKBONE.NAME}")
        model = load_clip_to_cpu(cfg)
        model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")

        tokens = _as_input_ids(tokenizer(prompts))  # (C, L)

        with torch.no_grad():
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", tf, persistent=False)
        self.clip_model = model
        self.tokenizer = tokenizer

    def forward(self, image):
        device = image.device
        dt = next(self.clip_model.parameters()).dtype

        image_features = self.clip_model.encode_image(image.to(dtype=dt))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = getattr(self.clip_model, "logit_scale", None)
        if logit_scale is None:
            ls = torch.tensor(1 / 0.07, device=device, dtype=image_features.dtype)
        else:
            ls = logit_scale.exp().to(device=device, dtype=image_features.dtype)

        text_features = self.text_features.to(device=device, dtype=image_features.dtype)
        return ls * (image_features @ text_features.t())


class ConchCLIP_ZS(nn.Module):
    def __init__(self, cfg, classnames, tokenizer):
        super().__init__()
        print("Loading Conch (open_clip_custom)")

        model = load_clip_to_cpu(cfg)
        model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")

        # IMPORTANT: conch tokenizer supporte très bien padding/max_length
        batch = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = _as_input_ids(batch)  # (C, 77)

        with torch.no_grad():
            tf = model.encode_text(input_ids)
            tf = tf / tf.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", tf, persistent=False)
        self.clip_model = model
        self.tokenizer = tokenizer

    def forward(self, image):
        device = image.device
        dt = next(self.clip_model.parameters()).dtype

        image_features = self.clip_model.encode_image(image.to(dtype=dt))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = getattr(self.clip_model, "logit_scale", None)
        if logit_scale is None:
            ls = torch.tensor(1 / 0.07, device=device, dtype=image_features.dtype)
        else:
            ls = logit_scale.exp().to(device=device, dtype=image_features.dtype)

        text_features = self.text_features.to(device=device, dtype=image_features.dtype)
        return ls * (image_features @ text_features.t())


class PubMedCLIP_ZS(nn.Module):
    def __init__(self, cfg, classnames, tokenizer):
        super().__init__()
        print("Loading PubMedCLIP (HF)")

        model = load_clip_to_cpu(cfg)
        model.float()
        self.clip_model = model

        # ✅ Freeze tout de suite (inclut logit_scale)
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]

        tok = tokenizer(
            prompts_,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        # device est CPU ici, mais c'est OK : register_buffer + .to(device) du module déplacera le buffer
        device = next(self.clip_model.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.no_grad():
            tf = self.clip_model.get_text_features(
                input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
            )
            tf = tf / tf.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", tf, persistent=False)
        self.vision_dtype = next(self.clip_model.vision_model.parameters()).dtype

    @torch.no_grad()
    def forward(self, image):
        device = image.device

        image_features = self.clip_model.get_image_features(
            pixel_values=image.to(dtype=self.vision_dtype)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_features.to(device=device, dtype=image_features.dtype)

        logit_scale = getattr(self.clip_model, "logit_scale", None)
        if logit_scale is None:
            ls = torch.tensor(1 / 0.07, device=device, dtype=image_features.dtype)
        else:
            ls = logit_scale.exp().to(device=device, dtype=image_features.dtype)

        return ls * (image_features @ text_features.t())


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        vision_dtype = _vision_dtype_from_module(self.image_encoder)
        image_features = self.image_encoder(image.to(dtype=vision_dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(image.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().to(
            dtype=image_features.dtype, device=image.device
        )
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomBiomedCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.biomed = biomed_model
        self.tokenizer = tokenizer

        self.image_encoder = self.biomed.encode_image
        self.vision = self.biomed.visual

        self.text = self.biomed.text
        self.word_embeddings = self.text.transformer.embeddings.word_embeddings

        hidden = self.word_embeddings.weight.shape[1]
        self.prompt_learner = BiomedPromptLearner(
            cfg, classnames, biomed_model, hidden, tokenizer, self.word_embeddings
        )

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = getattr(self.biomed, "logit_scale", None)

    def encode_text_with_ctx(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (n_cls, L)
        device = input_ids.device
        we = self.word_embeddings

        ctx = self.prompt_learner.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)

        ctx = ctx.to(device=device, dtype=we.weight.dtype)
        n_ctx = ctx.size(1)

        def _inject_ctx(module, inp, out):
            # out: (B, L, H)
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx.to(device=out.device, dtype=out.dtype)
            return out

        h = we.register_forward_hook(_inject_ctx)
        try:
            # open_clip CustomTextCLIP -> encode_text fait: BERT -> pooler -> proj
            tf = self.biomed.encode_text(input_ids)
        finally:
            h.remove()

        return tf

    def forward(self, image):
        device = image.device

        vision_dtype = _vision_dtype_from_module(self.vision)
        image_features = self.image_encoder(image.to(dtype=vision_dtype))

        input_ids = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(input_ids)

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
        self.clip_model = clip_model
        self.prompt_learner = QuiltPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.dtype = next(self.image_encoder.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts):

        te = _get_openclip_token_embedding(self.clip_model)
        ctx = self.prompt_learner.ctx
        if ctx.dim() == 2:
            ctx_for_classes = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)
        else:
            print("CSC type but not wanted")

        ctx_for_classes = ctx_for_classes.to(
            device=tokenized_prompts.device, dtype=te.weight.dtype
        )
        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(
                device=out.device, dtype=out.dtype
            )
            return out

        h = te.register_forward_hook(_inject_ctx)
        try:
            return self.clip_model.encode_text(tokenized_prompts)
        finally:
            h.remove()

    def forward(self, image):
        device = image.device
        image_features = self.image_encoder(image.to(dtype=self.dtype))

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)

        # normalize
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
        self.clip_model = conch_model
        self.tokenizer = tokenizer
        self.prompt_learner = ConchPromptLearner(
            cfg, classnames, conch_model, tokenizer
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = getattr(conch_model, "logit_scale", None)
        self.dtype = next(self.clip_model.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts):
        te = _get_openclip_token_embedding(self.clip_model)
        ctx = self.prompt_learner.ctx
        if ctx.dim() == 2:
            ctx_for_classes = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)
        else:
            print("CSC type but not wanted")
        ctx_for_classes = ctx_for_classes.to(
            device=tokenized_prompts.device, dtype=te.weight.dtype
        )
        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(
                device=out.device, dtype=out.dtype
            )
            return out

        h = te.register_forward_hook(_inject_ctx)
        try:
            return self.clip_model.encode_text(tokenized_prompts)
        finally:
            h.remove()

    def forward(self, image):
        device = image.device
        image_features = self.clip_model.encode_image(image.to(dtype=self.dtype))

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)

        # normalize + logits
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ---- logit scale ----
        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=text_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=text_features.dtype
            )

        return logit_scale * image_features @ text_features.t()


class CustomPubMedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.model = clip_model
        self.tokenizer = tokenizer
        self.prompt_learner = HFPromptLearner(cfg, classnames, clip_model, tokenizer)

        self.logit_scale = clip_model.logit_scale

        # pour hook
        self.token_embedding = clip_model.text_model.embeddings.token_embedding

    def forward(self, image):
        device = image.device

        # -----------------------
        # 1) image features
        # -----------------------
        vision_dtype = next(self.model.vision_model.parameters()).dtype
        imf = self.model.get_image_features(pixel_values=image.to(dtype=vision_dtype))
        imf = imf / imf.norm(dim=-1, keepdim=True)

        # -----------------------
        # 2) build prompt embeds (CoOp-style)
        # -----------------------
        input_ids = self.prompt_learner.tokenized_prompts.to(device)
        attention_mask = self.prompt_learner.attention_mask.to(device)
        prompt_embeds = self.prompt_learner().to(device=device)  # (C, 77, H)

        # -----------------------
        # 3) hook token_embedding to override embeddings
        # -----------------------
        def _override_token_embedding(module, inp, out):
            # out: (C, 77, H)
            return prompt_embeds.to(dtype=out.dtype, device=out.device)

        h = self.token_embedding.register_forward_hook(_override_token_embedding)
        try:
            tf = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        finally:
            h.remove()

        tf = tf / tf.norm(dim=-1, keepdim=True)

        # -----------------------
        # 4) logits
        # -----------------------
        logit_scale = self.logit_scale.exp().to(device=device, dtype=imf.dtype)
        return logit_scale * (imf @ tf.t())


class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T, -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss


@TRAINER_REGISTRY.register()
class ProGrad(TrainerX):
    """Projected Gradient for few-shot CLIP"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        backbone = cfg.MODEL.BACKBONE.NAME

        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if cfg.MODEL.BACKBONE.NAME == "BiomedCLIP":
            tokenizer = get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            print("Building zeroshot BiomedCLIP")
            self.zs_clip = BiomedCLIP_ZS(cfg, classnames, tokenizer)

            print("Building custom BiomedCLIP")
            self.model = CustomBiomedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Quilt-B/32":
            tokenizer = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")
            print("Building zeroshot Quilt-B/32")
            self.zs_clip = QuiltCLIP_ZS(cfg, classnames, tokenizer)

            print("Building custom Quilt-B/16")
            self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Quilt-B/16":
            tokenizer = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-16")
            print("Building zeroshot Quilt-B/32")
            self.zs_clip = QuiltCLIP_ZS(cfg, classnames, tokenizer)

            print("Building custom Quilt-B/32")
            self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "PubMedCLIP-B/32":
            tokenizer = CLIPTokenizerFast.from_pretrained(
                "flaviagiammarino/pubmed-clip-vit-base-patch32"
            )
            print("Building zeroshot PubMedCLIP")
            self.zs_clip = PubMedCLIP_ZS(cfg, classnames, tokenizer)

            print("Building custom PubMedCLIP")
            self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "PLIP-B/32":
            tokenizer = CLIPTokenizerFast.from_pretrained("vinid/plip")
            print("Building zeroshot PLIP")
            self.zs_clip = PubMedCLIP_ZS(cfg, classnames, tokenizer)

            print("Building custom PLIP")
            self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Conch":
            tokenizer = conch.open_clip_custom.get_tokenizer()
            print("Building zeroshot Conch")
            self.zs_clip = ConchCLIP_ZS(cfg, classnames, tokenizer)

            print("Building custom Conch")
            self.model = CustomConchCLIP(cfg, classnames, clip_model, tokenizer)

        else:
            print("Building zeroshot CLIP")
            self.zs_clip = CLIP_ZS(cfg, classnames)

            print("Building custom CLIP")
            self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in ZS Clip model")
        for _, param in self.zs_clip.named_parameters():
            param.requires_grad_(False)

        print("Turning off gradients in CoOp model")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None:
                print("[DEBUG] grad is None for:", name, p.shape, p.device, p.dtype)

        print("✅ Model built successfully")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # ✅ freeze avant sanity check (même si déjà fait dans la classe, ça sécurise)
        for p in self.zs_clip.parameters():
            p.requires_grad_(False)

        self.model.to(self.device)
        self.zs_clip = self.zs_clip.cuda()

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.zs_clip = nn.DataParallel(self.zs_clip)

        # build criterion
        if cfg.LOSS.NAME == "prograd":
            self.criterion = ProGradLoss(T=cfg.LOSS.T)
        else:
            raise NotImplementedError

        sanity_check_prograd_all_backbones(
            self,
            image_size=cfg.INPUT.SIZE[0],
            batch_size=2,
            verbose=True,
            require_projection_effect=False,
        )

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                with torch.no_grad():
                    zs_clip_output = self.zs_clip(image)
                loss = self.criterion(output, zs_clip_output.detach(), label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            with torch.no_grad():
                zs_clip_output = self.zs_clip(image)

            xe_loss, kl_loss = self.criterion(output, zs_clip_output.detach(), label)
            self.prograd_backward_and_update(xe_loss, kl_loss, self.cfg.LOSS.LAMBDA)

        loss_summary = {
            "xe_loss": xe_loss.item(),
            "kl_loss": kl_loss.item(),
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
