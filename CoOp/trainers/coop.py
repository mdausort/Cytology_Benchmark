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


def _vision_dtype_from_module(vision: torch.nn.Module) -> torch.dtype:
    """
    Retrieve the dtype used by the vision encoder.
    Arg:
        vision: vision encoder module.
    Return:
        dtype: dtype of the vision module parameters.
    """
    if hasattr(vision, "conv1") and hasattr(vision.conv1, "weight"):
        return vision.conv1.weight.dtype
    return next(vision.parameters()).dtype


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
    raise AttributeError("Could not find token_embedding in this Quilt/open_clip model")


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
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
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

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        """
        Build the complete prompts by combining prefix, learnable context, and suffix tokens.
        Return:
            prompts: prompt embeddings for all classes.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
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
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = word_embeddings.weight.dtype
        self.csc = cfg.TRAINER.COOP.CSC

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
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, batch_size, device, dtype):
        """
        Build the complete prompts by combining prefix, learnable context, and suffix tokens.
        Arg:
            batch_size: number of prompt sets to generate.
            device: target device.
            dtype: target dtype.
        Return:
            prompts: prompt embeddings.
        """
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

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

        dtype = next(quilt_model.parameters()).dtype

        token_embedding = _get_openclip_token_embedding(quilt_model)
        ctx_dim = token_embedding.weight.shape[1]

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

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        """
        Build the complete prompts by combining prefix, learnable context, and suffix tokens.
        Return:
            prompts: prompt embeddings for all classes.
        """
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

        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = cfg.TRAINER.COOP.N_CTX
        self.csc = cfg.TRAINER.COOP.CSC

        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        dtype = next(conch_model.parameters()).dtype

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

        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        """
        Build the complete prompts by combining prefix, learnable context, and suffix tokens.
        Return:
            prompts: prompt embeddings for all classes.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
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
            ]

            with torch.no_grad():
                emb = self.token_embedding(tok)

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
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        """
        Build the complete prompts by combining prefix, learnable context, and suffix tokens.
        Return:
            prompts: prompt embeddings for all classes.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        return torch.cat([prefix, ctx, suffix], dim=1)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        """
        Initialize the CoOp model based on the original CLIP backbone.
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

    def forward(self, image):
        """
        Compute classification logits from image features and prompted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
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
        """
        Initialize the CoOp model based on BiomedCLIP.
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

    def encode_text_with_ctx(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text while injecting the learnable context into the token embeddings.
        Arg:
            input_ids: tokenized text inputs.
        Return:
            tf: encoded text features.
        """
        device = input_ids.device
        we = self.word_embeddings

        ctx = self.prompt_learner.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)

        ctx = ctx.to(device=device, dtype=we.weight.dtype)
        n_ctx = ctx.size(1)

        def _inject_ctx(module, inp, out):
            out = out.clone()
            out[:, 1 : 1 + n_ctx, :] = ctx.to(device=out.device, dtype=out.dtype)
            return out

        h = we.register_forward_hook(_inject_ctx)
        try:
            tf = self.biomed.encode_text(input_ids)
        finally:
            h.remove()

        return tf

    def forward(self, image):
        """
        Compute classification logits from image features and prompted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
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
        """
        Initialize the CoOp model based on a Quilt/OpenCLIP backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Quilt/OpenCLIP backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = QuiltPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.dtype = next(self.image_encoder.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts):
        """
        Encode tokenized prompts while replacing the context token embeddings with the learnable prompt vectors.
        Arg:
            tokenized_prompts: tokenized prompts.
        Return:
            text_features: encoded text features.
        """
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
        """
        Compute classification logits from image features and prompted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
        device = image.device
        image_features = self.image_encoder(image.to(dtype=self.dtype))

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()
        return logits


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        """
        Initialize the CoOp model based on the Conch backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            conch_model: Conch backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
        """
        Encode tokenized prompts while replacing the context token embeddings with the learnable prompt vectors.
        Arg:
            tokenized_prompts: tokenized prompts.
        Return:
            text_features: encoded text features.
        """
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
        """
        Compute classification logits from image features and prompted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
        device = image.device
        image_features = self.clip_model.encode_image(image.to(dtype=self.dtype))

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
        """
        Initialize the CoOp model based on a Hugging Face CLIP-based model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Hugging Face CLIP-based backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer

        self.prompt_learner = HFPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.attention_mask = self.prompt_learner.attention_mask

        self.token_embedding = clip_model.text_model.embeddings.token_embedding
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.vision_dtype = next(self.clip_model.vision_model.parameters()).dtype

    def encode_text_with_ctx(self, tokenized_prompts):
        """
        Encode tokenized prompts while replacing the context token embeddings with the learnable prompt vectors.
        Arg:
            tokenized_prompts: tokenized prompts.
        Return:
            text_features: encoded text features.
        """
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
            tf = self.clip_model.get_text_features(
                input_ids=tokenized_prompts, attention_mask=attention_mask
            )
        finally:
            h.remove()

        return tf

    def forward(self, image):
        """
        Compute classification logits from image features and prompted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
        device = next(self.clip_model.parameters()).device
        image_features = self.clip_model.get_image_features(
            pixel_values=image.to(dtype=self.vision_dtype)
        )
        tokenized_prompts = self.tokenized_prompts.to(device)
        self.attention_mask = self.attention_mask.to(device)
        text_features = self.encode_text_with_ctx(tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
