import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX  # type: ignore
from dassl.metrics import compute_accuracy  # type: ignore
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


def _get_vision_weight_ref(m: nn.Module) -> torch.Tensor:
    """
    Retourne un poids de référence du backbone vision (pour dtype/device),
    quel que soit le wrapper (OpenAI, open_clip, HF, etc.).
    """
    # 1) OpenAI CLIP / Quilt open_clip: image_encoder est un nn.Module (visual)
    if hasattr(m, "image_encoder") and isinstance(m.image_encoder, nn.Module):
        enc = m.image_encoder
        if hasattr(enc, "conv1") and hasattr(enc.conv1, "weight"):
            return enc.conv1.weight
        return next(enc.parameters())

    # 2) BiomedCLIP open_clip: encode_image est une fonction, vision est dans biomed.visual
    if hasattr(m, "biomed"):
        vision = getattr(m.biomed, "visual", None)
        if isinstance(vision, nn.Module):
            if hasattr(vision, "conv1") and hasattr(vision.conv1, "weight"):
                return vision.conv1.weight
            return next(vision.parameters())

    # 3) Conch/open_clip: clip_model a souvent .visual
    if hasattr(m, "clip_model"):
        cm = m.clip_model
        vision = getattr(cm, "visual", None)
        if isinstance(vision, nn.Module):
            if hasattr(vision, "conv1") and hasattr(vision.conv1, "weight"):
                return vision.conv1.weight
            return next(vision.parameters())
        # HF transformers CLIP: vision_model
        vision = getattr(cm, "vision_model", None)
        if isinstance(vision, nn.Module):
            return next(vision.parameters())

    raise RuntimeError("Could not find a vision module to infer dtype/device.")


def debug_one_update_step(trainer, batch):
    import torch
    from torch.nn import functional as F

    model = trainer.model
    model.train()

    # mini optim juste pour le sanity check (n'affecte pas trainer.optim)
    optim = torch.optim.SGD(model.prompt_learner.parameters(), lr=1e-1)

    img, label = trainer.parse_batch_train(batch)
    with torch.no_grad():
        m = model.module if hasattr(model, "module") else model
        vision = getattr(m, "image_encoder", None)
        if vision is not None and hasattr(vision, "conv1"):
            img = img.to(dtype=vision.conv1.weight.dtype)

    # snapshot ctx avant
    ctx0 = model.prompt_learner.ctx.detach().clone()

    # forward/backward
    optim.zero_grad(set_to_none=True)
    out = model(img)
    loss = F.cross_entropy(out, label)
    loss.backward()

    # stats grad
    g = model.prompt_learner.ctx.grad
    g_mean = None if g is None else g.abs().mean().item()
    g_max = None if g is None else g.abs().max().item()

    # step
    optim.step()

    # snapshot ctx après
    ctx1 = model.prompt_learner.ctx.detach().clone()

    # mesure changement
    delta = (ctx1 - ctx0).abs()
    print("\n" + "=" * 80)
    print("🧪 ONE UPDATE STEP CHECK")
    print("loss:", float(loss.item()))
    print("grad ctx mean/max:", g_mean, g_max)
    print("ctx change mean/max:", delta.mean().item(), delta.max().item())
    print("ctx norm before/after:", ctx0.norm().item(), ctx1.norm().item())

    # assertions robustes
    assert g is not None and g.abs().sum().item() > 0, "❌ ctx grad is zero/None"
    assert delta.sum().item() > 0, "❌ ctx did not change after optimizer.step()"
    print("✅ ctx updates correctly")
    print("=" * 80 + "\n")


def debug_sanity_check(trainer, batch):
    import torch
    from torch.nn import functional as F

    print("\n" + "=" * 80)
    print("🔍 SANITY CHECK CoOp")
    print("=" * 80)

    model = trainer.model

    # --------------------------------------------------
    # 1) Backbone
    # --------------------------------------------------
    print("\n[1] Model type")
    print("Model class:", type(model))
    is_biomed = isinstance(model, CustomBiomedCLIP)
    is_quilt = isinstance(model, CustomQuiltCLIP)
    is_pubmed = isinstance(model, CustomPubMedCLIP)
    is_conch = isinstance(model, CustomConchCLIP)
    is_openai = isinstance(model, CustomCLIP)

    if is_biomed:
        print("→ Using BiomedCLIP backbone")
    elif is_quilt:
        print("→ Using Quilt/OpenCLIP backbone")
    elif is_pubmed:
        print("→ Using PubMedCLIP (HF transformers) backbone")
    elif is_conch:
        print("→ Using Conch backbone")
    else:
        print("→ Using OpenAI CLIP backbone")

    # --------------------------------------------------
    # 2) Trainable params
    # --------------------------------------------------
    print("\n[2] Trainable parameters")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"Trainable params ({len(trainable)}):")
    for n in trainable:
        print("  ✔", n)
    print(f"Frozen params ({len(frozen)}): (not listed, OK)")

    assert all(
        "prompt_learner" in n for n in trainable
    ), "❌ ERROR: non-prompt parameters are trainable!"

    # --------------------------------------------------
    # 3) logit_scale
    # --------------------------------------------------
    print("\n[3] logit_scale")
    logit_scale = getattr(model, "logit_scale", None)
    print("logit_scale:", logit_scale)
    if logit_scale is not None:
        try:
            print("  value:", logit_scale.exp().item())
        except Exception:
            pass

    # --------------------------------------------------
    # 4) Forward / grads sanity
    # --------------------------------------------------
    print("\n[4] Forward pass")
    model.zero_grad(set_to_none=True)
    model.train()

    img, label = trainer.parse_batch_train(batch)

    with torch.no_grad():
        m = model.module if hasattr(model, "module") else model
        w = _get_vision_weight_ref(m)
        print(
            "[SANITY] img:",
            img.device,
            img.dtype,
            "| vision weight:",
            w.device,
            w.dtype,
        )
        img = img.to(device=w.device, dtype=w.dtype)

    # --- Token ids pour classes (n_cls, L)
    tok = None
    if hasattr(model, "tokenized_prompts"):
        tok = model.tokenized_prompts.to(img.device)

    # === TEXT PATH CHECK ===
    if is_quilt:
        # ✅ chemin réel Quilt/OpenCLIP: hook injection
        tf = model.encode_text_with_ctx(tok)
        print("tf_hook.requires_grad:", tf.requires_grad)

        g = torch.autograd.grad(
            tf.sum(), model.prompt_learner.ctx, retain_graph=True, allow_unused=True
        )[0]
        print("g is None:", g is None)
        if g is not None:
            print("g absmean/max:", g.abs().mean().item(), g.abs().max().item())

    elif is_openai:
        # ✅ chemin OpenAI CLIP (text_encoder existe)
        prompts = model.prompt_learner()
        tf = model.text_encoder(prompts, tok)
        print("tf_custom.requires_grad:", tf.requires_grad)

        g = torch.autograd.grad(
            tf.sum(), model.prompt_learner.ctx, retain_graph=True, allow_unused=True
        )[0]
        print("g is None:", g is None)
        if g is not None:
            print("g absmean/max:", g.abs().mean().item(), g.abs().max().item())

    elif is_biomed:
        # BiomedCLIP: ton text_encoder prend des tokenized dict/tensor,
        # et ctx est injecté via inputs_embeds.
        ids = model.tokenized_prompts.to(img.device)
        tf = model.encode_text_with_ctx(ids)
        print("biomed tf:", tf.shape, tf.dtype, tf.device)

        g = torch.autograd.grad(
            tf.sum(), model.prompt_learner.ctx, retain_graph=True, allow_unused=True
        )[0]

        print("g is None:", g is None)
        if g is not None:
            print("g absmean/max:", g.abs().mean().item(), g.abs().max().item())

    elif is_pubmed:
        # ✅ CoOp-like: on encode le texte à partir des prompts embeddings
        tf = model.encode_text_with_ctx(tok)  # (n_cls, dim)
        print("tf_pubmed.requires_grad:", tf.requires_grad)
        print("tf shape:", tf.shape)

        if tf.size(0) >= 2:
            print(
                "cos(tf0, tf1):",
                torch.nn.functional.cosine_similarity(tf[0:1], tf[1:2]).item(),
            )
            print("l2(tf0-tf1):", (tf[0] - tf[1]).norm().item())

        # token ids des prompts (pour debug)
        ids = model.tokenized_prompts.to(img.device)  # (n_cls, L)
        am = model.attention_mask.to(img.device)  # (n_cls, L)
        for i in range(min(2, ids.size(0))):
            L = int(am[i].sum().item())
            print(
                i,
                "decoded:",
                repr(model.tokenizer.decode(ids[i][:L], skip_special_tokens=False)),
            )

        g = torch.autograd.grad(
            tf.sum(), model.prompt_learner.ctx, retain_graph=True, allow_unused=True
        )[0]
        print("g is None:", g is None)
        if g is not None:
            print("g absmean/max:", g.abs().mean().item(), g.abs().max().item())

    elif is_conch:
        tok = model.tokenized_prompts.to(img.device)
        tf = model.encode_text_with_ctx(tok)
        print("tf_conch.requires_grad:", tf.requires_grad)
        g = torch.autograd.grad(
            tf.sum(), model.prompt_learner.ctx, retain_graph=True, allow_unused=True
        )[0]
        print("g is None:", g is None)
        if g is not None:
            print("g absmean/max:", g.abs().mean().item(), g.abs().max().item())

    # === LOGITS SENSITIVITY CHECK ===
    out1 = model(img)

    with torch.no_grad():
        model.prompt_learner.ctx.add_(0.1 * torch.randn_like(model.prompt_learner.ctx))

    out2 = model(img)
    print("mean|logits2-logits1|:", (out2 - out1).abs().mean().item())

    print("Logits:", out1.shape, out1.dtype, out1.device)
    assert out1.shape[0] == img.shape[0], "❌ Batch size mismatch"
    assert out1.shape[1] == trainer.dm.num_classes, "❌ Num classes mismatch"

    # --------------------------------------------------
    # 5) Loss + backward
    # --------------------------------------------------
    print("\n[5] Loss + backward")
    loss = F.cross_entropy(out1, label)
    print("Loss:", loss.item())
    loss.backward()

    # --------------------------------------------------
    # 6) Gradients check
    # --------------------------------------------------
    print("\n[6] Gradient check (should be ONLY prompt_learner)")
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                print(f"  {name}: grad None ❌")
            else:
                print(
                    f"  {name}: grad mean={p.grad.abs().mean():.6f} max={p.grad.abs().max():.6e}"
                )
        else:
            if p.grad is not None:
                print(f"❌ ERROR: frozen param has grad → {name}")

    print("\n✅ SANITY CHECK PASSED")
    print("=" * 80 + "\n")


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
    raise AttributeError("Could not find token_embedding in this Quilt/open_clip model")


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

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

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

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

        dtype = next(quilt_model.parameters()).dtype

        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = token_embedding.weight.shape[1]  # robuste pour Quilt/open_clip

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
        self.tokenizer = tokenizer
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

        self.hidden = clip_model.text_model.config.hidden_size
        self.token_embedding = clip_model.text_model.embeddings.token_embedding

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
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        return torch.cat([prefix, ctx, suffix], dim=1)


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

    def encode_text_with_ctx(self, tokenized_prompts):
        device = next(self.clip_model.parameters()).device
        tokenized_prompts = tokenized_prompts.to(device)
        attention_mask = self.attention_mask.to(device)

        ctx = self.prompt_learner.ctx
        if ctx.dim() == 2:
            ctx_for_classes = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)
        else:
            print("CSC type but not wanted")
        ctx_for_classes = ctx_for_classes.to(
            device=device, dtype=self.token_embedding.weight.dtype
        )

        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
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

    def forward(self, image):
        device = next(self.clip_model.parameters()).device
        image_features = self.clip_model.get_image_features(
            pixel_values=image.to(dtype=self.vision_dtype)
        )
        tokenized_prompts = self.tokenized_prompts.to(device)
        self.attention_mask = self.attention_mask.to(device)
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


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
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

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Count of parameters
        m = self.model.module if hasattr(self.model, "module") else self.model
        tot, tr = count_params(m)
        print(f"[PARAMS] total={tot:,} | trainable={tr:,} | trainable%={100*tr/tot:.4f}%")

        pl = count_params_by_module(m, "prompt_learner")
        if pl is not None:
            print(f"[PARAMS] prompt_learner total/trainable = {pl[0]:,} / {pl[1]:,}")

        # Debug - sanity check
        batch_debug = next(iter(self.train_loader_x))
        debug_sanity_check(self, batch_debug)
        debug_one_update_step(self, batch_debug)

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

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
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
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
