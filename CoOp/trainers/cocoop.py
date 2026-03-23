import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX  # type: ignore
from dassl.utils import load_pretrained_weights, load_checkpoint  # type: ignore
from dassl.optim import build_optimizer, build_lr_scheduler  # type: ignore

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer  # type: ignore
from open_clip import get_tokenizer, create_model_from_pretrained
from transformers import CLIPModel, CLIPTokenizerFast
import conch.open_clip_custom


_tokenizer = _Tokenizer()


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_params_by_module(model: nn.Module, key="prompt_learner"):
    # utile pour vérifier que seul prompt_learner est entraîné
    sub = dict(model.named_modules()).get(key, None)
    if sub is None:
        return None
    total = sum(p.numel() for p in sub.parameters())
    trainable = sum(p.numel() for p in sub.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def infer_vis_dim_from_encode_image(
    biomed_model: nn.Module, image_size: int = 224
) -> int:
    biomed_model.eval()

    # sauvegarde device/dtype
    p = next(biomed_model.parameters())
    orig_device = p.device
    orig_dtype = p.dtype

    # on infère sur CPU pour éviter les OOM / allocations GPU au build
    biomed_model_cpu = biomed_model.to(device="cpu")

    x = torch.zeros(1, 3, image_size, image_size, device="cpu", dtype=torch.float32)

    # certains modèles aiment float16, mais CPU fp16 peut casser -> on tente fp32 d'abord
    feats = biomed_model_cpu.encode_image(x)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    vis_dim = int(feats.shape[-1])

    # restore
    biomed_model.to(device=orig_device, dtype=orig_dtype)
    return vis_dim


def _get_image_features_only(m, img):
    """
    Calcul image_features (B,D) normalisés en respectant le dtype du backbone.
    Important: éviter le mismatch fp16/fp32.
    """
    kind = _infer_wrapper_kind(m)

    if kind == "biomed":
        dt = next(m.biomed.parameters()).dtype
        feats = m.biomed.encode_image(img.to(dtype=dt))

    elif kind == "hf":
        dt = next(m.clip_model.vision_model.parameters()).dtype
        feats = m.clip_model.get_image_features(pixel_values=img.to(dtype=dt))

    elif kind == "quilt":
        dt = next(m.clip_model.parameters()).dtype
        feats = m.clip_model.encode_image(img.to(dtype=dt))

    elif kind == "conch":
        dt = next(m.clip_model.parameters()).dtype
        feats = m._encode_image(img.to(dtype=dt))

    elif kind == "openai":
        # ⚠️ OpenAI CLIP: utilise m.dtype (fiable) plutôt que next(parameters())
        dt = getattr(m, "dtype", None)
        if dt is None:
            dt = next(m.image_encoder.parameters()).dtype
        feats = m.image_encoder(img.to(dtype=dt))

    else:
        raise RuntimeError(
            f"Could not infer how to compute image features (wrapper={kind})."
        )

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _infer_wrapper_kind(m):
    # 1) Biomed (ton wrapper)
    if hasattr(m, "biomed") and (
        hasattr(m, "encode_text_with_ctx")
        or hasattr(m, "word_embeddings")
        or hasattr(m, "tokenizer")
    ):
        return "biomed"

    # 2) HF wrapper (si tu l'appelles self.clip_model ou self.model, adapte ici)
    if (
        hasattr(m, "clip_model")
        and hasattr(m.clip_model, "get_image_features")
        and hasattr(m.clip_model, "get_text_features")
    ):
        return "hf"

    # 3) Conch (plus spécifique que quilt si tu as _encode_image)
    if (
        hasattr(m, "clip_model")
        and hasattr(m, "_encode_image")
        and hasattr(m, "encode_text_with_ctx")
    ):
        return "conch"

    # 4) Quilt (open_clip)
    if (
        hasattr(m, "clip_model")
        and hasattr(m, "encode_text_with_ctx")
        and hasattr(m, "tokenized_prompts")
    ):
        return "quilt"

    # 5) OpenAI CLIP
    if (
        hasattr(m, "image_encoder")
        and hasattr(m, "text_encoder")
        and hasattr(m, "tokenized_prompts")
    ):
        return "openai"

    return "unknown"


def _get_ctx_shifted_and_bias(m, image_features):
    """
    Retourne:
      ctx_shifted: (B, n_ctx, H)
      bias: (B, H) (peut être None)
      ctx_base: base ctx param (pour debug)

    IMPORTANT: Ne doit pas appeler pl(image_features) si pl n'a pas de forward custom.
    """
    pl = m.prompt_learner
    kind = _infer_wrapper_kind(m)

    # HF: utilise API native ctx_shifted()
    if hasattr(pl, "ctx_shifted") and callable(getattr(pl, "ctx_shifted")):
        ctx_shifted = pl.ctx_shifted(image_features)
        meta_dtype = next(pl.meta_net.parameters()).dtype
        bias = pl.meta_net(image_features.to(dtype=meta_dtype))
        return ctx_shifted, bias, pl.ctx

    # Biomed: ton wrapper a souvent _ctx_shifted (sur m)
    if (
        kind == "biomed"
        and hasattr(m, "_ctx_shifted")
        and callable(getattr(m, "_ctx_shifted"))
    ):
        ctx_shifted = m._ctx_shifted(image_features)
        meta_dtype = next(m.prompt_learner.meta_net.parameters()).dtype
        bias = m.prompt_learner.meta_net(image_features.to(dtype=meta_dtype))
        return ctx_shifted, bias, m.prompt_learner.ctx

    # Quilt/Conch: prompt_learner.forward retourne (B,n_ctx,H)
    # => on l'appelle seulement si forward est surchargé
    has_custom_forward = pl.__class__.forward is not nn.Module.forward
    if has_custom_forward:
        out = pl(image_features)
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            ctx_shifted = out
            meta_dtype = next(pl.meta_net.parameters()).dtype
            bias = pl.meta_net(image_features.to(dtype=meta_dtype))
            return ctx_shifted, bias, pl.ctx

    # Fallback "OpenAI-like": ctx + meta_net
    if hasattr(pl, "meta_net") and hasattr(pl, "ctx"):
        meta_dtype = next(pl.meta_net.parameters()).dtype
        bias = pl.meta_net(image_features.to(dtype=meta_dtype))  # (B,H)
        ctx_base = pl.ctx.to(dtype=meta_dtype)  # (n_ctx,H) ou (1,n_ctx,H)
        if ctx_base.dim() == 2:
            ctx_shifted = ctx_base.unsqueeze(0) + bias.unsqueeze(1)  # (B,n_ctx,H)
        elif ctx_base.dim() == 3:
            # si déjà (1,n_ctx,H)
            ctx_shifted = ctx_base + bias.unsqueeze(1)
        else:
            raise RuntimeError(f"Unexpected ctx dim: {ctx_base.dim()}")
        return ctx_shifted, bias, pl.ctx

    raise RuntimeError(
        f"Could not obtain ctx_shifted/bias for prompt_learner type={type(pl)} (wrapper={kind})."
    )


def _compute_logits_from_ctx_swap(m, img, image_features, ctx_shifted_swapped):
    """
    Reconstruit les logits en injectant un ctx_shifted (B,n_ctx,H) "swapped".
    Retourne swapped logits (B,n_cls) ou None si pas supporté.
    """
    device = img.device
    kind = _infer_wrapper_kind(m)

    B = img.size(0)
    if B < 2:
        return None

    if kind == "quilt":
        tok = m.tokenized_prompts.to(device)
        n_cls = tok.size(0)
        logits = []
        for b in range(B):
            ctx_b = ctx_shifted_swapped[b].unsqueeze(0).expand(n_cls, -1, -1)
            tf = m.encode_text_with_ctx(tok, ctx_b)
            tf = tf / tf.norm(dim=-1, keepdim=True)

            ls = (
                m.logit_scale.exp().to(device=device, dtype=tf.dtype)
                if m.logit_scale is not None
                else torch.tensor(1 / 0.07, device=device, dtype=tf.dtype)
            )
            imf = image_features[b].to(dtype=tf.dtype)
            logits.append(ls * (imf @ tf.t()))
        return torch.stack(logits, dim=0)

    if kind == "conch":
        tok = m.tokenized_prompts.to(device)
        n_cls = tok.size(0)
        logits = []
        for b in range(B):
            ctx_b = ctx_shifted_swapped[b].unsqueeze(0).expand(n_cls, -1, -1)
            tf = m.encode_text_with_ctx(tok, ctx_b)
            tf = tf / tf.norm(dim=-1, keepdim=True)

            ls = (
                m.logit_scale.exp().to(device=device, dtype=tf.dtype)
                if m.logit_scale is not None
                else torch.tensor(1 / 0.07, device=device, dtype=tf.dtype)
            )
            imf = image_features[b].to(dtype=tf.dtype)
            logits.append(ls * (imf @ tf.t()))
        return torch.stack(logits, dim=0)

    if kind == "hf":
        device = img.device

        tok = m.tokenized_prompts.to(device)  # (n_cls, L)
        # selon ton wrapper, l'attention_mask est soit sur m soit sur prompt_learner
        am = (m.attention_mask if hasattr(m, "attention_mask") else m.prompt_learner.attention_mask).to(device)

        n_cls = tok.size(0)
        logits = []

        for b in range(B):
            # ctx_shifted_swapped: (B, n_ctx, hidden)
            ctx_b = ctx_shifted_swapped[b].unsqueeze(0).expand(n_cls, -1, -1)  # (n_cls, n_ctx, hidden)

            tf = m.encode_text_with_ctx(tok, am, ctx_b)  # (n_cls, D)
            tf = tf / tf.norm(dim=-1, keepdim=True)

            ls = (
                m.logit_scale.exp().to(device=device, dtype=tf.dtype)
                if getattr(m, "logit_scale", None) is not None
                else torch.tensor(1 / 0.07, device=device, dtype=tf.dtype)
            )

            imf = image_features[b].to(dtype=tf.dtype)
            logits.append(ls * (imf @ tf.t()))

        return torch.stack(logits, dim=0)

    if kind == "biomed":
        device = img.device
        input_ids = m.tokenized_prompts.to(device)  # (n_cls, L)
        n_cls = input_ids.shape[0]

        logits = []
        for b in range(B):
            ctx_b = ctx_shifted_swapped[b].unsqueeze(0).expand(n_cls, -1, -1)  # (n_cls, n_ctx, H)

            tf = m.encode_text_with_ctx(input_ids, ctx_b)  # ✅ (n_cls, D)
            tf = tf / tf.norm(dim=-1, keepdim=True)

            ls = (
                m.logit_scale.exp().to(device=device, dtype=tf.dtype)
                if getattr(m, "logit_scale", None) is not None
                else torch.tensor(1 / 0.07, device=device, dtype=tf.dtype)
            )
            imf = image_features[b].to(dtype=tf.dtype)
            logits.append(ls * (imf @ tf.t()))

        return torch.stack(logits, dim=0)

    if kind == "openai":
        tokenized_prompts = m.tokenized_prompts.to(device)
        prefix = m.prompt_learner.token_prefix
        suffix = m.prompt_learner.token_suffix
        n_cls = m.prompt_learner.n_cls

        def _construct_for_classes(ctx_for_classes):
            return torch.cat([prefix, ctx_for_classes, suffix], dim=1)

        logits = []
        for b in range(B):
            ctx_b = ctx_shifted_swapped[b].unsqueeze(0).expand(n_cls, -1, -1)
            pts = _construct_for_classes(ctx_b)
            tf = m.text_encoder(pts, tokenized_prompts)
            tf = tf / tf.norm(dim=-1, keepdim=True)

            ls = m.logit_scale.exp().to(device=device, dtype=tf.dtype)
            imf = image_features[b].to(dtype=tf.dtype)
            logits.append(ls * (imf @ tf.t()))
        return torch.stack(logits, dim=0)

    return None


def debug_sanity_check(trainer, batch):
    print("\n" + "=" * 100)
    print("🔍 SANITY CHECK CoCoOp (ROBUST, MULTI-BACKBONE)")
    print("=" * 100)

    model = trainer.model
    img, label = trainer.parse_batch_train(batch)

    # unwrap DataParallel
    m = model.module if hasattr(model, "module") else model
    kind = _infer_wrapper_kind(m)

    print("\n[0] Batch")
    print("  img:", tuple(img.shape), img.dtype, img.device)
    print(
        "  label:",
        tuple(label.shape),
        label.dtype,
        label.device,
        "min/max:",
        int(label.min().item()),
        int(label.max().item()),
    )

    print("\n[1] Model")
    print("  model class:", type(m))
    print("  wrapper kind:", kind)
    print("  training:", m.training)

    print("\n[2] Trainable parameters (should be ONLY prompt_learner.*)")
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    print(f"  n_trainable: {len(trainable)}")
    for n, p in trainable:
        print(f"   ✔ {n:55s} shape={tuple(p.shape)} dtype={p.dtype} device={p.device}")

    names = [n for n, _ in trainable]
    assert any(
        n.endswith(".ctx") and "prompt_learner" in n for n in names
    ), "❌ ctx not trainable"
    assert any(
        "prompt_learner.meta_net" in n for n in names
    ), "❌ meta_net not trainable"
    assert all("prompt_learner" in n for n in names), "❌ non-prompt params trainable"

    print("\n[3] logit_scale")
    logit_scale = getattr(m, "logit_scale", None)
    print("  logit_scale:", logit_scale)
    if logit_scale is not None:
        try:
            print("  exp(logit_scale):", float(logit_scale.exp().item()))
            print("  dtype/device:", logit_scale.dtype, logit_scale.device)
        except Exception as e:
            print("  (could not print exp):", repr(e))

    # [4] Forward eval -> logits
    print("\n[4] Forward eval -> logits")
    was_training = m.training
    m.eval()
    with torch.no_grad():
        out = m(img)
    print(
        "  logits:",
        tuple(out.shape),
        out.dtype,
        out.device,
        "min/max:",
        float(out.min().item()),
        float(out.max().item()),
    )
    m.train(was_training)

    # [5] Sensitivity to ctx (global perturb)
    print("\n[5] Sensitivity to ctx (eval) [global ctx perturb]")
    m.eval()
    with torch.no_grad():
        out1 = m(img)

        ctx0 = m.prompt_learner.ctx.detach().clone()
        eps = 0.01 * torch.randn_like(ctx0)
        m.prompt_learner.ctx.copy_(ctx0 + eps)

        out2 = m(img)
        m.prompt_learner.ctx.copy_(ctx0)

    diff = (out2 - out1).abs().mean().item()
    print("  mean|out2-out1|:", float(diff))
    assert diff > 0, "❌ logits not sensitive to ctx (suspicious)"

    # [6] Visual-conditioning check: ctx_shifted differs across images
    print("\n[6] Visual-conditioning check (ctx_shifted & bias vary across images)")
    m.eval()
    with torch.no_grad():
        image_features = _get_image_features_only(m, img)
        ctx_shifted, bias, _ = _get_ctx_shifted_and_bias(m, image_features)

    print("  image_features:", tuple(image_features.shape), image_features.dtype)
    print("  ctx_shifted:", tuple(ctx_shifted.shape), ctx_shifted.dtype)
    if bias is not None:
        print("  bias:", tuple(bias.shape), bias.dtype)

    if img.size(0) >= 2:
        d_ctx = (ctx_shifted[0] - ctx_shifted[1]).abs().mean().item()
        print("  mean|ctx_shifted[0]-ctx_shifted[1]|:", float(d_ctx))
        assert (
            d_ctx > 0
        ), "❌ ctx_shifted identical for different images (conditioning broken?)"
        if bias is not None:
            d_b = (bias[0] - bias[1]).abs().mean().item()
            print("  mean|bias[0]-bias[1]|:", float(d_b))
            assert (
                d_b > 0
            ), "❌ bias identical for different images (meta_net not conditioning?)"
    else:
        print("  (batch size < 2, skip pairwise conditioning checks)")

    # [7] Swap test
    print("\n[7] Swap test (ctx conditioning affects logits)")
    if img.size(0) >= 2:
        m.eval()
        with torch.no_grad():
            base = m(img)

            # swap ctx_shifted[0] <-> ctx_shifted[1]
            ctx_sw = ctx_shifted.clone()
            ctx_sw[0], ctx_sw[1] = ctx_shifted[1].clone(), ctx_shifted[0].clone()

            swapped = _compute_logits_from_ctx_swap(m, img, image_features, ctx_sw)

        if swapped is None:
            print("  (swap test skipped: wrapper not supported)")
        else:
            d_swap = (swapped - base).abs().mean().item()
            print("  mean|logits_swapped - logits_base|:", float(d_swap))
            assert (
                d_swap > 0
            ), "❌ swapping ctx did not change logits (conditioning not used?)"
    else:
        print("  (batch size < 2, skip swap test)")

    # [8] Train forward -> loss + backward (grads exist)
    print("\n[8] Train forward -> loss + backward (grads)")
    m.train()
    m.zero_grad(set_to_none=True)
    loss = m(img, label)
    print("  loss:", float(loss.item()), "dtype:", loss.dtype, "device:", loss.device)
    loss.backward()

    ctx_grad = m.prompt_learner.ctx.grad
    # meta_net linear2 peut ne pas exister si tu as renommé
    w_grad = None
    if hasattr(m.prompt_learner, "meta_net") and hasattr(
        m.prompt_learner.meta_net, "linear2"
    ):
        w_grad = m.prompt_learner.meta_net.linear2.weight.grad

    print("  ctx.grad is None?", ctx_grad is None)
    if ctx_grad is not None:
        print(
            "  ctx.grad abs mean/max:",
            float(ctx_grad.abs().mean().item()),
            float(ctx_grad.abs().max().item()),
        )
        assert ctx_grad.abs().mean().item() > 0, "❌ ctx.grad is ~0"

    if w_grad is not None:
        print("  meta_net.linear2.weight.grad is None?", w_grad is None)
        if w_grad is not None:
            print(
                "  meta_net.linear2.weight.grad abs mean/max:",
                float(w_grad.abs().mean().item()),
                float(w_grad.abs().max().item()),
            )
            assert w_grad.abs().mean().item() > 0, "❌ meta_net grad is ~0"
    else:
        print("  (meta_net.linear2 not found, skip that specific grad check)")

    print("\n[9] Gradients (trainable only)")
    for n, p in trainable:
        g = p.grad
        if g is None:
            print(f"   ❌ {n}: grad None")
        else:
            print(
                f"   ✔ {n:55s} grad mean={g.abs().mean().item():.6e} "
                f"max={g.abs().max().item():.6e} dtype={g.dtype}"
            )

    print("\n✅ SANITY CHECK DONE")
    print("=" * 100 + "\n")


def _to_device_tokenized(tokenized, device):

    if isinstance(tokenized, torch.Tensor):
        return tokenized.to(device)
    if isinstance(tokenized, dict):
        return {
            k: v.to(device) if torch.is_tensor(v) else torch.as_tensor(v).to(device)
            for k, v in tokenized.items()
        }
    raise TypeError(f"Unexpected tokenized type: {type(tokenized)}")


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


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
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

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        )  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

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

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class BiomedPromptLearner(nn.Module):
    def __init__(
        self, cfg, classnames, biomed_model, hidden_size, tokenizer, word_embeddings
    ):
        super().__init__()
        device = next(biomed_model.parameters()).device

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = word_embeddings.weight.dtype

        vis_dim = None

        # 1) heuristiques rapides (si dispo)
        vision = getattr(biomed_model, "visual", None)
        if vis_dim is None and vision is not None:
            if hasattr(vision, "output_dim"):
                vis_dim = int(vision.output_dim)
            elif hasattr(vision, "proj") and hasattr(vision.proj, "weight"):
                # cas fréquent: proj: Linear(in_features=?, out_features=embed_dim)
                vis_dim = int(vision.proj.weight.shape[0])

        if vis_dim is None and hasattr(biomed_model, "embed_dim"):
            vis_dim = int(biomed_model.embed_dim)

        # 2) fallback ultime: forward CPU sur encode_image
        if vis_dim is None:
            vis_dim = infer_vis_dim_from_encode_image(
                biomed_model, image_size=cfg.INPUT.SIZE[0]
            )

        assert vis_dim is not None, "Could not infer vis_dim for Biomed meta_net"

        # -------------------------
        # 1) Init ctx (CTX_INIT ou random)
        # -------------------------
        if ctx_init:
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
            ctx_vectors = torch.empty(n_ctx, hidden_size, dtype=dtype)
            prompt_prefix = " ".join(["X"] * n_ctx)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context tokens: {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, hidden_size)),
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class QuiltPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        n_ctx = cfg.TRAINER.COCOOP.N_CTX

        dtype = next(quilt_model.parameters()).dtype

        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = token_embedding.weight.shape[1]  # robuste pour Quilt/open_clip

        vis_dim = None
        if hasattr(quilt_model, "visual") and hasattr(quilt_model.visual, "output_dim"):
            vis_dim = quilt_model.visual.output_dim
        if vis_dim is None and hasattr(quilt_model, "embed_dim"):
            vis_dim = quilt_model.embed_dim
        if (
            vis_dim is None
            and hasattr(quilt_model, "config")
            and hasattr(quilt_model.config, "embed_dim")
        ):
            vis_dim = quilt_model.config.embed_dim
        assert vis_dim is not None, "Could not infer vis_dim for Conch meta_net"

        # -------------------------
        # 1) Init ctx (CTX_INIT ou random)
        # -------------------------
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init

            tok = tokenizer([ctx_init])
            if isinstance(tok, dict):
                tok = torch.as_tensor(tok.get("input_ids", tok[list(tok.keys())[0]]))
            else:
                tok = torch.as_tensor(tok)

            with torch.no_grad():
                emb = token_embedding(tok).type(dtype)  # (1, L, dim)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()

        else:
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(ctx_vectors)

        # meta-net: image_features -> bias (dim ctx)
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class ConchPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        self.tokenizer = tokenizer

        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        n_ctx = cfg.TRAINER.COCOOP.N_CTX

        # dim texte Conch = 768 (d'après ton print)
        ctx_dim = conch_model.text.ln_final.weight.shape[
            0
        ]  # pt ctx_dim = conch_model.text.token_embedding.weight.shape[1]
        dtype = next(conch_model.parameters()).dtype

        # vis_dim pour meta_net
        vis_dim = None
        if hasattr(conch_model, "visual") and hasattr(conch_model.visual, "output_dim"):
            vis_dim = conch_model.visual.output_dim
        if vis_dim is None and hasattr(conch_model, "embed_dim"):
            vis_dim = conch_model.embed_dim
        if (
            vis_dim is None
            and hasattr(conch_model, "config")
            and hasattr(conch_model.config, "embed_dim")
        ):
            vis_dim = conch_model.config.embed_dim
        assert vis_dim is not None, "Could not infer vis_dim for Conch meta_net"

        # -------------------------
        # 1) init ctx
        # -------------------------
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init

            tok = tokenizer(
                [ctx_init],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )[
                "input_ids"
            ]  # (1, 77)

            with torch.no_grad():
                emb = conch_model.text.token_embedding(tok).type(dtype)  # (1, 77, dim)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()  # après BOS

        else:
            n_ctx = n_ctx
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, dim)

        # meta-net
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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
        )[
            "input_ids"
        ]  # (n_cls, 77)

        self.tokenized_prompts = tokenized

        with torch.no_grad():
            embedding = conch_model.text.token_embedding(self.tokenized_prompts).type(
                dtype
            )

        self.register_buffer("token_prefix", embedding[:, :1, :])  # BOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )  # le reste

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

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class HFPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        n_ctx = cfg.TRAINER.COCOOP.N_CTX

        self.token_embedding = clip_model.text_model.embeddings.token_embedding

        # dims
        self.hidden = clip_model.text_model.config.hidden_size  # ex: 512
        self.vis_dim = clip_model.config.projection_dim  # ex: 512

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init

            tok = tokenizer(
                [ctx_init],
                padding=False,
                truncation=True,
                return_tensors="pt",
            )[
                "input_ids"
            ]  # (1, L)

            with torch.no_grad():
                emb = self.token_embedding(tok)  # (1, L, hidden)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()

        else:
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.empty(n_ctx, self.hidden)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, hidden)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.vis_dim, self.vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(self.vis_dim // 16, self.hidden)),
                ]
            )
        )

        # précision
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tok_full = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        self.tokenized_prompts = tok_full["input_ids"]  # (n_cls, 77)
        self.attention_mask = tok_full["attention_mask"]  # (n_cls, 77)

        with torch.no_grad():
            embedding = self.token_embedding(self.tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx

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

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


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
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

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

    def encode_text_with_ctx(
        self, input_ids: torch.Tensor, ctx_for_classes: torch.Tensor
    ):
        # input_ids: (n_cls, L)
        # ctx_for_classes: (n_cls, n_ctx, 768)
        we = self.word_embeddings
        n_ctx = ctx_for_classes.size(1)

        ctx_for_classes = ctx_for_classes.to(
            device=input_ids.device, dtype=we.weight.dtype
        )

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(
                device=out.device, dtype=out.dtype
            )
            return out

        h = we.register_forward_hook(_inject_ctx)
        try:
            return self.biomed.encode_text(input_ids)
        finally:
            h.remove()

    def forward(self, image, label=None):
        device = image.device
        dtype = next(self.biomed.parameters()).dtype

        tokenized_prompts = self.tokenized_prompts.to(device)  # (n_cls, L)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)
        else:
            logit_scale = self.logit_scale.exp().to(device=device)

        image_features = self.image_encoder(image.to(dtype=dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        n_ctx = self.prompt_learner.n_ctx
        logits = []

        for pts_i, imf_i in zip(prompts, image_features):
            ctx_for_classes = pts_i[:, 1 : 1 + n_ctx, :]
            ctx_for_classes = ctx_for_classes.to(device=device, dtype=dtype)
            text_features = self.encode_text_with_ctx(
                tokenized_prompts, ctx_for_classes
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale.to(text_features.dtype) * (
                imf_i.to(text_features.dtype) @ text_features.t()
            )
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


class CustomQuiltCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.prompt_learner = QuiltPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.dtype = next(self.clip_model.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts, ctx_for_classes):

        te = _get_openclip_token_embedding(self.clip_model)
        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(out.dtype)
            return out

        h = te.register_forward_hook(_inject_ctx)
        try:
            return self.clip_model.encode_text(tokenized_prompts)
        finally:
            h.remove()

    def forward(self, image, label=None):
        device = image.device

        tokenized_prompts = self.tokenized_prompts.to(device)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.to(dtype=self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        n_ctx = self.prompt_learner.n_ctx
        te = _get_openclip_token_embedding(self.clip_model)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            ctx_for_classes = pts_i[:, 1 : 1 + n_ctx, :]
            ctx_for_classes = ctx_for_classes.to(device=device, dtype=te.weight.dtype)
            text_features = self.encode_text_with_ctx(
                tokenized_prompts, ctx_for_classes
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale.to(text_features.dtype) * (
                imf_i.to(text_features.dtype) @ text_features.t()
            )
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


class CustomPubMedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.tokenizer = tokenizer

        # prompt learner HF (celui que tu as déjà)
        self.prompt_learner = HFPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.attention_mask = self.prompt_learner.attention_mask

        # HF CLIP
        self.token_embedding = clip_model.text_model.embeddings.token_embedding
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        # dtype vision (souvent float32)
        self.vision_dtype = next(self.clip_model.vision_model.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts, attention_mask, ctx_for_classes):
        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
            # out: (n_cls, 77, hidden)
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(
                device=out.device, dtype=out.dtype
            )
            return out

        h = self.token_embedding.register_forward_hook(_inject_ctx)
        try:
            # HF CLIPModel: récupérer features texte "projetées" (dim CLIP)
            tf = self.clip_model.get_text_features(
                input_ids=tokenized_prompts, attention_mask=attention_mask
            )
        finally:
            h.remove()

        return tf

    def forward(self, image, label=None):
        device = image.device

        tokenized_prompts = self.tokenized_prompts.to(device)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        image_features = self.clip_model.get_image_features(
            pixel_values=image.to(dtype=self.vision_dtype)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        attention_mask = self.attention_mask.to(device)

        prompts = self.prompt_learner(image_features)
        n_ctx = self.prompt_learner.n_ctx

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            ctx_for_classes = pts_i[:, 1 : 1 + n_ctx, :]  # ✅ (n_cls, n_ctx, hidden)
            ctx_for_classes = ctx_for_classes.to(
                device=device, dtype=self.token_embedding.weight.dtype
            )

            text_features = self.encode_text_with_ctx(
                tokenized_prompts, attention_mask, ctx_for_classes
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            l_i = logit_scale.to(text_features.dtype) * (
                imf_i.to(text_features.dtype) @ text_features.t()
            )
            logits.append(l_i)

        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = conch_model
        self.tokenizer = tokenizer
        self.prompt_learner = ConchPromptLearner(
            cfg, classnames, conch_model, tokenizer
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = getattr(conch_model, "logit_scale", None)
        self.dtype = next(self.clip_model.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts, ctx_for_classes):
        te = _get_openclip_token_embedding(self.clip_model)
        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(out.dtype)
            return out

        h = te.register_forward_hook(_inject_ctx)
        try:
            return self.clip_model.encode_text(tokenized_prompts)
        finally:
            h.remove()

    def forward(self, image, label=None):
        device = image.device
        tokenized_prompts = self.tokenized_prompts.to(device)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)
        else:
            logit_scale = self.logit_scale.exp().to(device=device)

        image_features = self.clip_model.encode_image(image.to(dtype=self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        n_ctx = self.prompt_learner.n_ctx
        te = _get_openclip_token_embedding(self.clip_model)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            ctx_for_classes = pts_i[:, 1 : 1 + n_ctx, :]
            ctx_for_classes = ctx_for_classes.to(device=device, dtype=te.weight.dtype)
            text_features = self.encode_text_with_ctx(
                tokenized_prompts, ctx_for_classes
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale.to(text_features.dtype) * (
                imf_i.to(text_features.dtype) @ text_features.t()
            )
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
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

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Count of parameters
        m = self.model.module if hasattr(self.model, "module") else self.model
        tot, tr = count_params(m)
        print(
            f"[PARAMS] total={tot:,} | trainable={tr:,} | trainable%={100*tr/tot:.4f}%"
        )

        pl = count_params_by_module(m, "prompt_learner")
        if pl is not None:
            print(f"[PARAMS] prompt_learner total/trainable = {pl[0]:,} / {pl[1]:,}")

        # Debug - sanity check
        batch = next(iter(self.train_loader_x))
        debug_sanity_check(self, batch)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
