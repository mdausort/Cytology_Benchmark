import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from open_clip import get_tokenizer, create_model_from_pretrained
from transformers import CLIPModel, CLIPTokenizerFast
import conch.open_clip_custom


_tokenizer = _Tokenizer()


def count_params(model: nn.Module):
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


def count_params_by_module(model: nn.Module, key="prompt_learner"):
    """
    Count the total number of parameters and trainable parameters for a specific submodule.
    Arg:
        model: model containing the target submodule.
        key: name of the submodule.
    Return:
        total: total number of parameters in the submodule.
        trainable: number of trainable parameters in the submodule.
    """
    sub = dict(model.named_modules()).get(key, None)
    if sub is None:
        return None
    total = sum(p.numel() for p in sub.parameters())
    trainable = sum(p.numel() for p in sub.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def infer_vis_dim_from_encode_image(biomed_model: nn.Module, image_size: int = 224) -> int:
    """
    Infer the visual feature dimension by running the image encoder on a dummy input.
    Arg:
        biomed_model: model containing an image encoder.
        image_size: input image size used for the dummy forward pass.
    Return:
        vis_dim: inferred visual feature dimension.
    """
    biomed_model.eval()

    p = next(biomed_model.parameters())
    orig_device = p.device
    orig_dtype = p.dtype

    biomed_model_cpu = biomed_model.to(device="cpu")

    x = torch.zeros(1, 3, image_size, image_size, device="cpu", dtype=torch.float32)

    feats = biomed_model_cpu.encode_image(x)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    vis_dim = int(feats.shape[-1])

    biomed_model.to(device=orig_device, dtype=orig_dtype)
    return vis_dim


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
    raise AttributeError("Could not find token_embedding in open_clip model")


def load_clip_to_cpu(cfg):
    """
    Load the selected CLIP-like backbone on CPU.
    Arg:
        cfg: configuration object containing the backbone name.
    Return:
        model: loaded backbone model.
    """
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
        """
        Initialize the text encoder from the CLIP text backbone.
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
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Encode prompt embeddings into text features.
        Arg:
            prompts: embedded prompt tokens.
            tokenized_prompts: tokenized prompts used to identify the end-of-text token.
        Return:
            x: encoded text features.
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
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
        """
        Initialize the prompt learner for the original CLIP model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: CLIP backbone model.
        Return:
            None
        """
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
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
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

    def forward(self, im_features):
        """
        Generate instance-conditioned prompts from image features.
        Arg:
            im_features: image features used to condition the prompts.
        Return:
            prompts: prompt embeddings for each image and class.
        """
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class BiomedPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, hidden_size, tokenizer, word_embeddings):
        """
        Initialize the prompt learner for BiomedCLIP.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            biomed_model: BiomedCLIP model.
            hidden_size: text embedding dimension.
            tokenizer: tokenizer associated with the model.
            word_embeddings: word embedding layer of the text encoder.
        Return:
            None
        """
        super().__init__()
        device = next(biomed_model.parameters()).device

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = word_embeddings.weight.dtype

        vis_dim = None

        vision = getattr(biomed_model, "visual", None)
        if vis_dim is None and vision is not None:
            if hasattr(vision, "output_dim"):
                vis_dim = int(vision.output_dim)
            elif hasattr(vision, "proj") and hasattr(vision.proj, "weight"):
                vis_dim = int(vision.proj.weight.shape[0])

        if vis_dim is None and hasattr(biomed_model, "embed_dim"):
            vis_dim = int(biomed_model.embed_dim)

        if vis_dim is None:
            vis_dim = infer_vis_dim_from_encode_image(
                biomed_model, image_size=cfg.INPUT.SIZE[0]
            )

        assert vis_dim is not None, "Could not infer vis_dim for Biomed meta_net"

        if ctx_init:
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

        tokenized_prompts = tokenizer(prompts)
        if isinstance(tokenized_prompts, dict):
            tokenized_prompts = tokenized_prompts["input_ids"]
        tokenized_prompts = tokenized_prompts.to(device)

        self.tokenized_prompts = tokenized_prompts

        with torch.no_grad():
            embedding = word_embeddings(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts

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

    def forward(self, im_features):
        """
        Generate instance-conditioned prompts from image features.
        Arg:
            im_features: image features used to condition the prompts.
        Return:
            prompts: prompt embeddings for each image and class.
        """
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class QuiltPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, quilt_model, tokenizer):
        """
        Initialize the prompt learner for a Quilt/OpenCLIP model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            quilt_model: Quilt/OpenCLIP backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        n_ctx = cfg.TRAINER.COCOOP.N_CTX

        dtype = next(quilt_model.parameters()).dtype

        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = token_embedding.weight.shape[1]

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
                emb = token_embedding(tok).type(dtype)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()

        else:
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
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

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized = tokenizer(prompts)
        if isinstance(tokenized, dict):
            tokenized = torch.as_tensor(
                tokenized.get("input_ids", tokenized[list(tokenized.keys())[0]])
            )
        else:
            tokenized = torch.as_tensor(tokenized)

        self.tokenized_prompts = tokenized
        with torch.no_grad():
            embedding = token_embedding(self.tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

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

    def forward(self, im_features):
        """
        Generate instance-conditioned prompts from image features.
        Arg:
            im_features: image features used to condition the prompts.
        Return:
            prompts: prompt embeddings for each image and class.
        """
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class ConchPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        """
        Initialize the prompt learner for the Conch model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            conch_model: Conch backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        n_cls = len(classnames)
        self.tokenizer = tokenizer

        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        n_ctx = cfg.TRAINER.COCOOP.N_CTX

        ctx_dim = conch_model.text.ln_final.weight.shape[
            0
        ]
        dtype = next(conch_model.parameters()).dtype

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
            ]

            with torch.no_grad():
                emb = conch_model.text.token_embedding(tok).type(dtype)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()

        else:
            n_ctx = n_ctx
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
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
        ]

        self.tokenized_prompts = tokenized

        with torch.no_grad():
            embedding = conch_model.text.token_embedding(self.tokenized_prompts).type(
                dtype
            )

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )

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

    def forward(self, im_features):
        """
        Generate instance-conditioned prompts from image features.
        Arg:
            im_features: image features used to condition the prompts.
        Return:
            prompts: prompt embeddings for each image and class.
        """
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class HFPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        """
        Initialize the prompt learner for a Hugging Face CLIP-based model.
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
        n_cls = len(classnames)

        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        n_ctx = cfg.TRAINER.COCOOP.N_CTX

        self.token_embedding = clip_model.text_model.embeddings.token_embedding

        self.hidden = clip_model.text_model.config.hidden_size
        self.vis_dim = clip_model.config.projection_dim

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
            ]

            with torch.no_grad():
                emb = self.token_embedding(tok)

            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()

        else:
            prompt_prefix = " ".join(["X"] * n_ctx)
            ctx_vectors = torch.empty(n_ctx, self.hidden)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.vis_dim, self.vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(self.vis_dim // 16, self.hidden)),
                ]
            )
        )

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
        self.tokenized_prompts = tok_full["input_ids"]
        self.attention_mask = tok_full["attention_mask"]

        with torch.no_grad():
            embedding = self.token_embedding(self.tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx

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

    def forward(self, im_features):
        """
        Generate instance-conditioned prompts from image features.
        Arg:
            im_features: image features used to condition the prompts.
        Return:
            prompts: prompt embeddings for each image and class.
        """
        prefix = self.token_prefix
        suffix = self.token_suffix

        meta_dtype = next(self.meta_net.parameters()).dtype
        im_features = im_features.to(dtype=meta_dtype)

        ctx = self.ctx.to(dtype=meta_dtype)
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        """
        Initialize the CoCoOp model based on the original CLIP backbone.
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
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        """
        Compute logits or training loss from images using instance-conditioned prompts.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
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
        """
        Initialize the CoCoOp model based on BiomedCLIP.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            biomed_model: BiomedCLIP backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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

    def encode_text_with_ctx(self, input_ids: torch.Tensor, ctx_for_classes: torch.Tensor):
        """
        Encode text features while injecting class-specific context embeddings.
        Arg:
            input_ids: tokenized text inputs.
            ctx_for_classes: context embeddings for each class.
        Return:
            text_features: encoded text features.
        """
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
        """
        Compute logits or training loss from images using instance-conditioned prompts.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
        device = image.device
        dtype = next(self.biomed.parameters()).dtype

        tokenized_prompts = self.tokenized_prompts.to(device)

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
        """
        Initialize the CoCoOp model based on a Quilt/OpenCLIP backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Quilt/OpenCLIP backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
        """
        Encode text features while injecting class-specific context embeddings.
        Arg:
            tokenized_prompts: tokenized prompts.
            ctx_for_classes: context embeddings for each class.
        Return:
            text_features: encoded text features.
        """
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
        """
        Compute logits or training loss from images using instance-conditioned prompts.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
        device = image.device

        tokenized_prompts = self.tokenized_prompts.to(device)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)
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
        """
        Initialize the CoCoOp model based on a Hugging Face CLIP-based model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Hugging Face CLIP-based backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.tokenizer = tokenizer

        self.prompt_learner = HFPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.attention_mask = self.prompt_learner.attention_mask

        self.token_embedding = clip_model.text_model.embeddings.token_embedding
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.vision_dtype = next(self.clip_model.vision_model.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts, attention_mask, ctx_for_classes):
        """
        Encode text features while injecting class-specific context embeddings.
        Arg:
            tokenized_prompts: tokenized prompts.
            attention_mask: attention mask associated with the prompts.
            ctx_for_classes: context embeddings for each class.
        Return:
            text_features: encoded text features.
        """
        n_ctx = ctx_for_classes.size(1)

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx_for_classes.to(
                device=out.device, dtype=out.dtype
            )
            return out

        h = self.token_embedding.register_forward_hook(_inject_ctx)
        try:
            tf = self.clip_model.get_text_features(
                input_ids=tokenized_prompts, attention_mask=attention_mask
            )
        finally:
            h.remove()

        return tf

    def forward(self, image, label=None):
        """
        Compute logits or training loss from images using instance-conditioned prompts.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
        device = image.device

        tokenized_prompts = self.tokenized_prompts.to(device)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)
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
            ctx_for_classes = pts_i[:, 1 : 1 + n_ctx, :]
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
        """
        Initialize the CoCoOp model based on the Conch backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            conch_model: Conch backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
        """
        Encode text features while injecting class-specific context embeddings.
        Arg:
            tokenized_prompts: tokenized prompts.
            ctx_for_classes: context embeddings for each class.
        Return:
            text_features: encoded text features.
        """
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
        """
        Compute logits or training loss from images using instance-conditioned prompts.
        Arg:
            image: input image batch.
            label: target labels used during training.
        Return:
            logits: classification logits during evaluation.
            loss: cross-entropy loss during training.
        """
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
