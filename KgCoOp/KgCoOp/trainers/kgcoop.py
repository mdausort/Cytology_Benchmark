import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from open_clip import get_tokenizer, create_model_from_pretrained
from transformers import CLIPModel, CLIPTokenizerFast
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
def vis_dim_from_encode_image_strict(model, image_size=224):
    was_training = model.training
    model.eval()

    p = next(model.parameters())
    orig_device, orig_dtype = p.device, p.dtype

    model = model.to("cpu")
    x = torch.zeros(1, 3, image_size, image_size, device="cpu", dtype=torch.float32)

    y = model.encode_image(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        # au cas où
        y = y.get("image_features", next(iter(y.values())))
    if y.dim() != 2:
        y = y.reshape(y.size(0), -1)

    d = int(y.shape[-1])

    model.to(device=orig_device, dtype=orig_dtype)
    model.train(was_training)
    return d


def _tokenize_to_ids(tokenizer, texts, device):
    tok = tokenizer(texts)

    # open_clip tokenizers: parfois dict, parfois Tensor, parfois list[list[int]]
    if isinstance(tok, dict):
        tok = tok.get("input_ids", next(iter(tok.values())))

    if hasattr(tok, "input_ids"):  # tokenizers.Encoding / BatchEncoding-like
        tok = tok.input_ids

    # si c'est déjà un tensor -> parfait
    if torch.is_tensor(tok):
        ids = tok.long()
    else:
        # list[list[int]] possiblement ragged -> pad à longueur max
        # (ou impose 77 si tu veux coller CLIP)
        max_len = max(len(x) for x in tok)
        ids = torch.zeros(len(tok), max_len, dtype=torch.long)
        for i, row in enumerate(tok):
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)

    return ids.to(device)


def sanity_check_kgcoop_safe(
    trainer, batch, w_score: float = 0.1, check_score_grad: bool = True
):
    import torch
    import torch.nn.functional as F

    print("\n" + "=" * 100)
    print("🔍 SANITY CHECK KgCoOp (SAFE, non-destructive)")
    print("=" * 100)

    model = trainer.model
    m = model.module if hasattr(model, "module") else model
    img, label = trainer.parse_batch_train(batch)

    # -------------------------------------------------
    # [1] Trainable params: ONLY ctx
    # -------------------------------------------------
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    names = [n for n, _ in trainable]
    print("[1] trainable:", names)
    assert len(trainable) > 0, "❌ no trainable params"
    assert all(("ctx" in n) for n in names), "❌ non-ctx param is trainable"

    # find ctx
    ctx_param, ctx_name = None, None
    for n, p in m.named_parameters():
        if n.endswith("prompt_learner.ctx") or n.endswith(".ctx") or n == "ctx":
            ctx_param, ctx_name = p, n
            break
    assert ctx_param is not None, "❌ cannot find ctx param"

    print(
        "[1] ctx param:",
        ctx_name,
        "shape:",
        tuple(ctx_param.shape),
        "dtype:",
        ctx_param.dtype,
        "device:",
        ctx_param.device,
    )

    # -------------------------------------------------
    # [2] Forward + backward once (CE + small score)
    # -------------------------------------------------
    m.train()

    # clear grads
    for p in m.parameters():
        if p.grad is not None:
            p.grad = None

    out, score = m(img)
    loss_ce = F.cross_entropy(out, label)
    total = loss_ce + (w_score * score if torch.is_tensor(score) else 0.0)

    total.backward()

    g = ctx_param.grad
    assert g is not None, "❌ ctx.grad is None"
    assert torch.isfinite(g).all(), "❌ ctx.grad has NaN/Inf"
    print("[2] logits:", tuple(out.shape), out.dtype, out.device)
    if torch.is_tensor(score):
        print("[2] score :", float(score.detach().item()))
    print(
        "[2] ctx.grad mean/max:",
        float(g.abs().mean().item()),
        float(g.abs().max().item()),
    )
    if float(g.abs().max().item()) == 0.0:
        print(
            "⚠ ctx.grad is all-zero after total.backward() (possible at init / degenerate batch)."
        )

    # -------------------------------------------------
    # [3] Optional: score -> ctx connectivity (NEW forward, no second backward on same graph)
    # -------------------------------------------------
    if check_score_grad:
        # IMPORTANT: new forward to create a fresh graph
        for p in m.parameters():
            if p.grad is not None:
                p.grad = None

        out2, score2 = m(img)

        if not torch.is_tensor(score2):
            raise AssertionError("❌ score is not a tensor")

        g_score = torch.autograd.grad(
            outputs=score2,
            inputs=ctx_param,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if g_score is None:
            raise AssertionError("❌ score is not connected to ctx (grad is None)")

        assert torch.isfinite(g_score).all(), "❌ grad(score->ctx) has NaN/Inf"
        print(
            "[3] grad(score->ctx) mean/max:",
            float(g_score.abs().mean().item()),
            float(g_score.abs().max().item()),
        )
        if float(g_score.abs().max().item()) == 0.0:
            print(
                "⚠ grad(score->ctx) is all-zero (often OK at init; connectivity is what matters)."
            )

    print("\n✅ SAFE KgCoOp sanity check done")
    print("=" * 100 + "\n")


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
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
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

        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(512, 512)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ("linear2", nn.Linear(128, 512))
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

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

        proj = biomed_model.text.proj
        D = None
        if hasattr(proj, "out_features"):
            D = int(proj.out_features)

        # cas 2: proj est un Sequential -> prendre le dernier Linear trouvé
        elif isinstance(proj, nn.Sequential):
            last_linear = None
            for mod in reversed(list(proj.modules())):
                if isinstance(mod, nn.Linear):
                    last_linear = mod
                    break
            if last_linear is None:
                raise AttributeError("biomed_model.text.proj is Sequential but contains no nn.Linear")
            D = int(last_linear.out_features)

        # cas 3: proj est un Parameter / Tensor / autre -> fallback shape
        elif hasattr(proj, "weight"):
            # ex: nn.Linear-like ou module custom
            D = int(proj.weight.shape[0])

        else:
            raise AttributeError(f"Cannot infer D from biomed_model.text.proj of type {type(proj)}")

        dtype = word_embeddings.weight.dtype
        self.csc = cfg.TRAINER.COOP.CSC

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
            vis_dim = vis_dim_from_encode_image_strict(
                biomed_model, image_size=cfg.INPUT.SIZE[0]
            )

        assert vis_dim is not None, "Could not infer vis_dim for Biomed meta_net"

        # -------------------------
        # 1) Init ctx (CTX_INIT ou random)
        # -------------------------
        if ctx_init:
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
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

        bias_vectors = torch.empty(1, D, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ("linear2", nn.Linear(128, 512))
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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
        self.classnames = classnames
        self.cfg = cfg
        self.clip_model = quilt_model
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX

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
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
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

            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(ctx_vectors)

        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        # -------------------------
        # 2) Construire prompts par classe
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")

        tok0 = tokenizer(prompts_)
        if isinstance(tok0, dict):
            ids0 = torch.as_tensor(tok0.get("input_ids", tok0[list(tok0.keys())[0]]))
        else:
            ids0 = torch.as_tensor(tok0)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(ids0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ("linear2", nn.Linear(128, 512))
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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
        self.register_buffer("ori_embeddings", text_features)
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
        self.classnames = classnames
        self.cfg = cfg
        n_cls = len(classnames)
        self.tokenizer = tokenizer

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

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
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
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

        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        # -------------------------
        # 2) tokenized prompts par classe
        # -------------------------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ("linear2", nn.Linear(128, 512))
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT

        self.csc = cfg.TRAINER.COOP.CSC

        self.token_embedding = clip_model.text_model.embeddings.token_embedding

        # dims
        self.hidden = clip_model.text_model.config.hidden_size  # ex: 512
        self.vis_dim = clip_model.config.projection_dim  # ex: 512

        dtype = next(clip_model.parameters()).dtype
        device = next(clip_model.parameters()).device

        if ctx_init:
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init

            tok = tokenizer(
                [ctx_init],
                padding=False,
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(
                device
            )  # (1, L)

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

        bias_vectors = torch.empty(1, self.hidden, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        print(f'HF Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # -------- 2) construire prompts string (EXACT CoOp) --------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.vis_dim, self.vis_dim)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ("linear2", nn.Linear(128, 512))
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

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


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.meta_net = self.prompt_learner.meta_net
        vis_dim = int(self.image_encoder.output_dim)
        self.adapter = Adapter(vis_dim, 4)

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def forward(self, image):
        prompts = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_old = self.ori_embedding

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(
            dim=-1, keepdim=True
        )
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


class CustomBiomedCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames

        self.biomed = biomed_model
        self.tokenizer = tokenizer
        self.classnames = classnames

        self.text = self.biomed.text
        self.word_embeddings = self.text.transformer.embeddings.word_embeddings

        hidden = self.word_embeddings.weight.shape[1]
        self.prompt_learner = BiomedPromptLearner(
            cfg, classnames, biomed_model, hidden, tokenizer, self.word_embeddings
        )

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = getattr(self.biomed, "logit_scale", None)
        self.dtype = next(self.biomed.parameters()).dtype
        self.meta_net = self.prompt_learner.meta_net

        vis_dim = vis_dim_from_encode_image_strict(
            self.biomed, image_size=cfg.INPUT.SIZE[0]
        )
        self.adapter = Adapter(vis_dim, 4)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]

        # --- encoder texte une fois pour l'ancre "old" ---
        device = next(self.biomed.parameters()).device
        input_ids = _tokenize_to_ids(tokenizer, prompts_, device)

        with torch.no_grad():
            tf_old = self.biomed.encode_text(input_ids)          # (n_cls, D) float/half
            tf_old = tf_old / tf_old.norm(dim=-1, keepdim=True)  # normalize

        self.register_buffer("ori_embedding", tf_old, persistent=False)

    def encode_image(self, image):
        dt = next(self.biomed.parameters()).dtype
        return self.biomed.encode_image(image.to(dtype=dt))

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
        dt = next(self.biomed.parameters()).dtype
        image_features = self.biomed.encode_image(image.to(dtype=dt))

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)
        text_features_old = self.ori_embedding.to(device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


class CustomQuiltCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = QuiltPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.dtype = next(self.clip_model.parameters()).dtype
        self.meta_net = self.prompt_learner.meta_net
        vis_dim = vis_dim_from_encode_image_strict(
            self.clip_model, image_size=cfg.INPUT.SIZE[0]
        )
        self.adapter = Adapter(vis_dim, 4)

        self.register_buffer(
            "ori_embedding", self.prompt_learner.ori_embeddings, persistent=False
        )

    def encode_image(self, image):
        return self.image_encoder(image.to(dtype=self.dtype))

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
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)
        text_features_old = self.ori_embedding.to(device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(
            dim=-1, keepdim=True
        )
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.clip_model = conch_model
        self.tokenizer = tokenizer
        self.classnames = classnames

        self.prompt_learner = ConchPromptLearner(
            cfg, classnames, conch_model, tokenizer
        )

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.encode_image
        self.logit_scale = getattr(self.clip_model, "logit_scale", None)
        self.dtype = next(self.clip_model.parameters()).dtype
        self.meta_net = self.prompt_learner.meta_net
        vis_dim = None
        if hasattr(conch_model, "visual") and hasattr(conch_model.visual, "output_dim"):
            vis_dim = int(conch_model.visual.output_dim)
        if vis_dim is None:
            vis_dim = vis_dim_from_encode_image_strict(
                conch_model, image_size=cfg.INPUT.SIZE[0]
            )

        self.adapter = Adapter(vis_dim, 4)

        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in self.classnames]

        device = next(self.clip_model.parameters()).device

        tok = tokenizer(
            prompts_,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(input_ids)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("ori_embedding", text_features, persistent=False)

    def encode_image(self, image):
        return self.image_encoder(image.to(dtype=self.dtype))

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
        text_features_old = self.ori_embedding.to(device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(
            dim=-1, keepdim=True
        )
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


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
        self.meta_net = self.prompt_learner.meta_net
        vis_dim = int(self.clip_model.config.projection_dim)
        self.adapter = Adapter(vis_dim, 4)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        tok = tokenizer(
            prompts_,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        device = next(self.clip_model.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.no_grad():
            tf = self.clip_model.get_text_features(
                input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
            )
            tf = tf / tf.norm(dim=-1, keepdim=True)
        self.register_buffer("ori_embedding", tf, persistent=False)

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

        text_features_old = self.ori_embedding.to(device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(
            dim=-1, keepdim=True
        )
        score = cos(text_features, text_features_old)
        score = 1.0 - torch.mean(score)

        return logits, score


@TRAINER_REGISTRY.register()
class KgCoOp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
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

        self.w = cfg.TRAINER.COOP.W

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # if "prompt_learner" not in name: # and "adapter" not in name:
            if "ctx" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # Count of parameters
        m = self.model.module if hasattr(self.model, "module") else self.model
        tot, tr = count_params(m)
        print(
            f"[PARAMS] total={tot:,} | trainable={tr:,} | trainable%={100*tr/tot:.4f}%"
        )

        pl = count_params_by_module(m, "ctx")
        if pl is not None:
            print(f"[PARAMS] ctx total/trainable = {pl[0]:,} / {pl[1]:,}")

        # Debug - sanity check
        batch_debug = next(iter(self.train_loader_x))
        sanity_check_kgcoop_safe(self, batch_debug)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        # self.optim_ = build_optimizer(self.model.adapter, cfg.OPTIM)
        # self.sched_ = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model('clip_adapter', self.model.adapter, self.optim_, self.sched_)

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
                output, score = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, score = self.model(image)
            loss = F.cross_entropy(output, label) + self.w * score
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            # self.update_lr()
            self.sched.step()
            # self.sched_.step()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input):
        return self.model(input)[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

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

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
