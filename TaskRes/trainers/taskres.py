"""
Task Residual Tuning
by Tao Yu (yutao666@mail.ustc.edu.cn)
Oct 4, 2022
"""

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


def _as_input_ids(tok_out):
    """
    Convert tokenizer outputs to a tensor of input ids.
    Arg:
        tok_out: tokenizer output in dict, tensor, or tokenizer-specific format.
    Return:
        input_ids: token ids as a tensor.
    """
    if isinstance(tok_out, dict):
        ids = tok_out.get("input_ids", list(tok_out.values())[0])
    else:
        ids = tok_out
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return torch.as_tensor(ids, dtype=torch.long)


def _extract_input_ids_any(tok, device, max_len=77, pad_id=0):
    """
    Extract input ids from various tokenizer output formats and move them to the target device.
    Arg:
        tok: tokenizer output.
        device: target device.
        max_len: target sequence length.
        pad_id: padding token id.
    Return:
        input_ids: padded or truncated token ids.
    """
    if torch.is_tensor(tok):
        ids = tok
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device=device)
        return _pad_trunc(ids.long(), max_len=max_len, pad_id=pad_id)

    if hasattr(tok, "data") and isinstance(tok.data, dict):
        tok = tok.data

    if isinstance(tok, dict):
        ids = tok.get("input_ids", None)
        if ids is None:
            ids = tok[next(iter(tok.keys()))]
        return _extract_input_ids_any(ids, device, max_len=max_len, pad_id=pad_id)

    if hasattr(tok, "ids") and isinstance(tok.ids, list):
        row = tok.ids[:max_len]
        if len(row) < max_len:
            row = row + [int(pad_id)] * (max_len - len(row))
        return torch.tensor([row], dtype=torch.long, device=device)

    if isinstance(tok, (list, tuple)) and len(tok) > 0 and hasattr(tok[0], "ids"):
        padded = []
        for enc in tok:
            row = enc.ids[:max_len]
            if len(row) < max_len:
                row = row + [int(pad_id)] * (max_len - len(row))
            padded.append(row)
        return torch.tensor(padded, dtype=torch.long, device=device)

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


def _pad_trunc(ids, max_len=77, pad_id=0):
    """
    Extract input ids from various tokenizer output formats and move them to the target device.
    Arg:
        tok: tokenizer output.
        device: target device.
        max_len: target sequence length.
        pad_id: padding token id.
    Return:
        input_ids: padded or truncated token ids.
    """
    if ids.size(1) > max_len:
        return ids[:, :max_len]
    if ids.size(1) < max_len:
        pad = ids.new_full((ids.size(0), max_len - ids.size(1)), int(pad_id))
        return torch.cat([ids, pad], dim=1)
    return ids


def _vision_dtype_from_module(vision):
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


# TaskRes(-Text)
class TaskResLearner(nn.Module):
    def __init__(self, cfg, base_text_features):
        """
        Initialize the task residual learner on top of base text features.
        Arg:
            cfg: configuration object.
            base_text_features: base text features used as initialization.
        Return:
            None
        """
        super().__init__()
        self.alpha = cfg.TRAINER.TaskRes.RESIDUAL_SCALE
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        """
        Compute residual-adjusted text features.
        Return:
            text_features: residual-adjusted text features.
        """
        return (
            self.base_text_features + self.alpha * self.text_feature_residuals
        )


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
    """
    Compute base text features for the original CLIP model from prompt templates.
    Arg:
        cfg: configuration object.
        classnames: list of class names.
        clip_model: CLIP backbone model.
        text_encoder: text encoder used to encode prompt embeddings.
    Return:
        text_embeddings: base text features.
    """
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
            )
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(
                    text_encoder(embeddings.cuda(), tokens.cuda())
                )
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


def _get_enhanced_base_text_features(
    cfg, classnames, clip_model, text_encoder, pretraiend_model
):
    """
    Compute enhanced base text features using a pretrained text projection.
    Arg:
        cfg: configuration object.
        classnames: list of class names.
        clip_model: CLIP backbone model.
        text_encoder: text encoder used to encode prompt embeddings.
        pretraiend_model: path to the pretrained text projection model.
    Return:
        text_embeddings: enhanced base text features.
    """
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
            )
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(
                    text_encoder(embeddings.cuda(), tokens.cuda())
                )
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)


def _get_base_text_features_biomed(cfg, classnames, biomed_model, tokenizer):
    """
    Compute base text features for BiomedCLIP from prompt templates.
    Arg:
        cfg: configuration object.
        classnames: list of class names.
        biomed_model: BiomedCLIP model.
        tokenizer: tokenizer associated with the model.
    Return:
        text_embeddings: base text features.
    """
    device = next(biomed_model.parameters()).device
    dtype = next(biomed_model.parameters()).dtype

    dataset = cfg.DATASET.NAME
    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for name in classnames:
            name = name.replace("_", " ")
            prompts_list = [template.format(name) for template in TEMPLATES]

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

            input_ids = _as_input_ids(tok).to(device)

            feats = biomed_model.encode_text(input_ids).to(
                dtype=dtype
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)

            text_embeddings.append(feats.mean(dim=0))

        text_embeddings = torch.stack(text_embeddings, dim=0)

    return text_embeddings


def _get_base_text_features_quilt(cfg, classnames, quilt_model, tokenizer):
    """
    Compute base text features for BiomedCLIP from prompt templates.
    Arg:
        cfg: configuration object.
        classnames: list of class names.
        biomed_model: BiomedCLIP model.
        tokenizer: tokenizer associated with the model.
    Return:
        text_embeddings: base text features.
    """
    device = next(quilt_model.parameters()).device
    dtype = next(quilt_model.parameters()).dtype

    dataset = cfg.DATASET.NAME
    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for name in classnames:
            name = name.replace("_", " ")

            prompts_list = [template.format(name) for template in TEMPLATES]

            tok = tokenizer(prompts_list)

            if isinstance(tok, dict):
                input_ids = tok.get("input_ids", list(tok.values())[0])
            else:
                input_ids = tok

            if hasattr(input_ids, "input_ids"):
                input_ids = input_ids.input_ids

            input_ids = torch.as_tensor(input_ids).to(device)

            feats = quilt_model.encode_text(input_ids).to(
                dtype=dtype
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)

            text_embeddings.append(feats.mean(dim=0))

        text_embeddings = torch.stack(text_embeddings, dim=0)

    return text_embeddings


def _get_base_text_features_conch(cfg, classnames, conch_model, tokenizer):
    """
    Compute base text features for the Conch model from prompt templates.
    Arg:
        cfg: configuration object.
        classnames: list of class names.
        conch_model: Conch model.
        tokenizer: tokenizer associated with the model.
    Return:
        text_embeddings: base text features.
    """
    device = next(conch_model.parameters()).device

    dataset = cfg.DATASET.NAME
    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]

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
            )
            feats = conch_model.encode_text(input_ids)

            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_embeddings.append(feats.mean(dim=0))

        text_embeddings = torch.stack(text_embeddings, dim=0)

    return text_embeddings


def _get_base_text_features_pubmedclip(cfg, classnames, clip_model, tokenizer):
    """
    Compute base text features for a Hugging Face CLIP-based model from prompt templates.
    Arg:
        cfg: configuration object.
        classnames: list of class names.
        clip_model: Hugging Face CLIP-based model.
        tokenizer: tokenizer associated with the model.
    Return:
        text_embeddings: base text features.
    """
    device = next(clip_model.parameters()).device
    dataset = cfg.DATASET.NAME

    TEMPLATES = [CUSTOM_TEMPLATES[dataset]]

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
            )

            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_embeddings.append(feats.mean(dim=0))
        text_embeddings = torch.stack(text_embeddings, dim=0)
    return text_embeddings


# TaskRes by Tao Yu, Oct 4, 2022
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        """
        Initialize the TaskRes model based on the original CLIP backbone.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: CLIP backbone model.
        Return:
            None
        """
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
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
        """
        Compute classification logits from image features and residual-adjusted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
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
        """
        Initialize the TaskRes model based on BiomedCLIP.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: BiomedCLIP backbone model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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

        self.prompt_learner = TaskResLearner(cfg, base_text_features)

    def forward(self, image):
        """
        Compute classification logits from image features and residual-adjusted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
        device = image.device
        vision_dtype = _vision_dtype_from_module(self.vision)
        try:
            image_features = self.image_encoder(image.to(dtype=vision_dtype))
        except:
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
        """
        Initialize the TaskRes model based on a Quilt/OpenCLIP backbone.
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

        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)

        self.dtype = next(self.image_encoder.parameters()).dtype
        base_text_features = _get_base_text_features_quilt(
            cfg, classnames, clip_model, tokenizer
        )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)

    def forward(self, image):
        """
        Compute classification logits from image features and residual-adjusted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
        device = image.device
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

        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)
        else:
            logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        """
        Initialize the TaskRes model based on the Conch backbone.
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

        base_text_features = _get_base_text_features_conch(
            cfg, classnames, conch_model, tokenizer
        )

        self.prompt_learner = TaskResLearner(cfg, base_text_features)
        self.logit_scale = getattr(conch_model, "logit_scale", None)
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, image):
        """
        Compute classification logits from image features and residual-adjusted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
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
        """
        Initialize the TaskRes model based on a Hugging Face CLIP-based model.
        Arg:
            cfg: configuration object.
            classnames: list of class names.
            clip_model: Hugging Face CLIP-based model.
            tokenizer: tokenizer associated with the model.
        Return:
            None
        """
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
        """
        Compute classification logits from image features and residual-adjusted text features.
        Arg:
            image: input image batch.
        Return:
            logits: similarity scores between image and text features.
        """
        device = image.device

        try:
            image_features = self.image_encoder(
                pixel_values=image.to(dtype=self.vision_dtype)
            )
        except Exception:
            image_features = self.image_encoder(pixel_values=image.float())

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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
