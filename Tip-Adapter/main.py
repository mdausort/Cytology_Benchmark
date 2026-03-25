import os
import sys
import random
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pytorch_warmup as warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import conch.open_clip_custom
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor
from open_clip import get_tokenizer, create_model_from_pretrained

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import (
    cls_acc,
    search_hp,
)


FINISH_MARKERS = [
    "Tip-Adapter-F's best test accuracy",
    "After fine-tuning, Tip-Adapter-F's best test accuracy",
]


def run_is_completed(log_dir):
    """
    Check whether a previous run has completed successfully based on log markers.
    Arg:
        log_dir: directory containing log files.
    Return:
        completed: whether the run is considered completed.
    """
    if not log_dir.exists():
        return False

    candidates = []
    for pat in ("*.out", "*.log", "*.txt"):
        candidates += list(log_dir.glob(pat))

    if not candidates:
        return False

    log_path = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000), 0)
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return False

    return any(m in tail for m in FINISH_MARKERS)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(cfg, shot, seed):
    """
    Set up logging to both console and file for the current run.
    Arg:
        cfg: configuration dictionary.
        shot: number of shots used in the experiment.
        seed: random seed.
    Return:
        None
    """
    dataset = cfg["dataset"]
    trainer = "Tip-Adapter"
    backbone = cfg["backbone"].replace("/", "")

    log_dir = Path(
        f"./Cytology_Benchmark/output/"
        f"{dataset}/{trainer}/{backbone}/{shot}shots/seed{seed}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "log.txt"
    f = open(log_file, "w")

    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = sys.stdout

    print("========================================")
    print("Logging initialized")
    print(f"Log file : {log_file}")
    print("========================================")


def get_arguments():
    """
    Parse command-line arguments.
    Return:
        args: parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", help="settings of Tip-Adapter in yaml format"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for reproducibility"
    )
    parser.add_argument("--shots", type=int, default=1, help="Shots")
    args = parser.parse_args()

    return args


def load_backbone_and_preprocess(cfg):
    """
    Load the selected backbone model together with its preprocessing pipeline.
    Arg:
        cfg: configuration dictionary.
    Return:
        model: loaded backbone model.
        preprocess: preprocessing function or transform.
        extra: dictionary containing tokenizer and backend metadata.
    """
    backbone = cfg["backbone"]

    if backbone == "Biomedclip":
        model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tok = get_tokenizer(model_id)
        return (
            model,
            preprocess,
            {"tokenizer": tok, "model_id": model_id, "kind": "openclip"},
        )

    elif backbone == "Quilt-B/16":
        model_id = "hf-hub:wisdomik/QuiltNet-B-16"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tok = get_tokenizer(model_id)
        return (
            model,
            preprocess,
            {"tokenizer": tok, "model_id": model_id, "kind": "openclip"},
        )

    elif backbone == "Quilt-B/32":
        model_id = "hf-hub:wisdomik/QuiltNet-B-32"
        model, preprocess = create_model_from_pretrained(model_id, device="cpu")
        tok = get_tokenizer(model_id)
        return (
            model,
            preprocess,
            {"tokenizer": tok, "model_id": model_id, "kind": "openclip"},
        )

    elif backbone == "Conch":
        model, preprocess = conch.open_clip_custom.create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=os.environ.get("HF_TOKEN", None),
        )
        tok = conch.open_clip_custom.get_tokenizer()
        return model, preprocess, {"tokenizer": tok, "kind": "conch"}

    elif backbone == "PubMedCLIP-B/32":
        model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id)
        tok = CLIPTokenizerFast.from_pretrained(model_id)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        return model, preprocess, {"tokenizer": tok, "model_id": model_id, "kind": "hf"}

    elif backbone == "PLIP-B/32":
        model_id = "vinid/plip"
        model = CLIPModel.from_pretrained(model_id)
        tok = CLIPTokenizerFast.from_pretrained(model_id)
        preprocess = CLIPProcessor.from_pretrained(model_id)
        return model, preprocess, {"tokenizer": tok, "model_id": model_id, "kind": "hf"}

    else:
        model_id = backbone
        clip_model, preprocess = clip.load(backbone)
        return clip_model, preprocess, {"kind": "clip"}


def clip_classifier(classnames, template, clip_model):
    """
    Build text classifier weights for the original CLIP model from prompt templates.
    Arg:
        classnames: list of class names.
        template: list of prompt templates.
        clip_model: CLIP backbone model.
    Return:
        clip_weights: normalized text classifier weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def biomedclip_classifier(classnames, templates, biomed_model, tokenizer, device=None):
    """
    Build text classifier weights for BiomedCLIP from prompt templates.
    Arg:
        classnames: list of class names.
        templates: list of prompt templates.
        biomed_model: BiomedCLIP model.
        tokenizer: tokenizer associated with the model.
        device: target device.
    Return:
        clip_weights: normalized text classifier weights.
    """
    if device is None:
        device = next(biomed_model.parameters()).device

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            tok = tokenizer(texts)

            if isinstance(tok, dict):
                input_ids = tok.get("input_ids", list(tok.values())[0])
                input_ids = torch.as_tensor(input_ids).to(device)
            else:
                input_ids = torch.as_tensor(tok).to(device)

            class_embeds = biomed_model.encode_text(input_ids)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)

    return clip_weights


def quilt_classifier(classnames, templates, quilt_model, tokenizer, device=None):
    """
    Build text classifier weights for a Quilt/OpenCLIP model from prompt templates.
    Arg:
        classnames: list of class names.
        templates: list of prompt templates.
        quilt_model: Quilt/OpenCLIP model.
        tokenizer: tokenizer associated with the model.
        device: target device.
    Return:
        clip_weights: normalized text classifier weights.
    """
    if device is None:
        device = next(quilt_model.parameters()).device

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            tok = tokenizer(texts)
            if isinstance(tok, dict):
                input_ids = tok.get("input_ids", list(tok.values())[0])
                input_ids = torch.as_tensor(input_ids).to(device)
            else:
                input_ids = torch.as_tensor(tok).to(device)

            class_embeds = quilt_model.encode_text(input_ids)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def conch_classifier(classnames, templates, conch_model, tokenizer, device=None):
    """
    Build text classifier weights for the Conch model from prompt templates.
    Arg:
        classnames: list of class names.
        templates: list of prompt templates.
        conch_model: Conch model.
        tokenizer: tokenizer associated with the model.
        device: target device.
    Return:
        clip_weights: normalized text classifier weights.
    """
    if device is None:
        device = next(conch_model.parameters()).device

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            tok = None
            try:
                tok = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
            except TypeError:
                tok = tokenizer(texts)

            if isinstance(tok, dict) or hasattr(tok, "input_ids"):
                input_ids = tok["input_ids"] if isinstance(tok, dict) else tok.input_ids
                input_ids = input_ids.to(device)
            elif (
                isinstance(tok, (list, tuple))
                and len(tok) > 0
                and hasattr(tok[0], "ids")
            ):
                input_ids = torch.tensor(
                    [enc.ids for enc in tok], device=device, dtype=torch.long
                )
            else:
                input_ids = torch.as_tensor(tok, device=device, dtype=torch.long)

            class_embeds = conch_model.encode_text(input_ids)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)

    return clip_weights


def pubmedclip_classifier(classnames, templates, clip_model, tokenizer, device=None):
    """
    Build text classifier weights for a Hugging Face CLIP-based model from prompt templates.
    Arg:
        classnames: list of class names.
        templates: list of prompt templates.
        clip_model: Hugging Face CLIP-based model.
        tokenizer: tokenizer associated with the model.
        device: target device.
    Return:
        clip_weights: normalized text classifier weights.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in templates]

            tok = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}

            class_embeds = clip_model.get_text_features(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
            )

            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            class_embed = class_embeds.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            clip_weights.append(class_embed)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache, shot):
    """
    Build or load the cache model for the original CLIP backbone.
    Arg:
        cfg: configuration dictionary.
        clip_model: CLIP backbone model.
        train_loader_cache: data loader used to build the cache.
        shot: number of shots used in the experiment.
    Return:
        cache_keys: cached visual features.
        cache_values: cached labels in one-hot format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = cfg["cache_dir"]
    keys_path = f"{cache_dir}/keys_{shot}shots.pt"
    values_path = f"{cache_dir}/values_{shot}shots.pt"

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []

                print(
                    "Augment Epoch: {:} / {:}".format(augment_idx, cfg["augment_epoch"])
                )
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.to(device)
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(device)
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys.cpu(), keys_path)
        torch.save(cache_values.cpu(), values_path)
    else:
        cache_keys = torch.load(keys_path, map_location=device)
        cache_values = torch.load(values_path, map_location=device)

    return cache_keys, cache_values


def build_cache_biomed(cfg, biomed_model, train_loader_cache, shot):
    """
    Build or load the cache model for BiomedCLIP or OpenCLIP-like backbones.
    Arg:
        cfg: configuration dictionary.
        biomed_model: backbone model used to encode images.
        train_loader_cache: data loader used to build the cache.
        shot: number of shots used in the experiment.
    Return:
        cache_keys: cached visual features.
        cache_values: cached labels in one-hot format.
    """
    cache_dir = cfg["cache_dir"]
    keys_path = f"{cache_dir}/keys_{shot}shots.pt"
    values_path = f"{cache_dir}/values_{shot}shots.pt"

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        biomed_model.eval()

        device = next(biomed_model.parameters()).device
        img_dtype = next(biomed_model.parameters()).dtype

        with torch.no_grad():
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []

                print(f"Augment Epoch: {augment_idx} / {cfg['augment_epoch']}")
                for images, target in tqdm(train_loader_cache):
                    images = images.to(device, non_blocking=True)

                    image_features = biomed_model.encode_image(
                        images.to(dtype=img_dtype)
                    )
                    train_features.append(image_features)

                    if augment_idx == 0:
                        cache_values.append(target.to(device, non_blocking=True))

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).contiguous()

        targets = torch.cat(cache_values, dim=0)
        num_classes = int(targets.max().item()) + 1
        cache_values = F.one_hot(targets, num_classes=num_classes).half()

        torch.save(cache_keys.cpu(), keys_path)
        torch.save(cache_values.cpu(), values_path)

    else:
        cache_keys = torch.load(keys_path, map_location=device)
        cache_values = torch.load(values_path, map_location=device)

    return cache_keys, cache_values


def build_cache_pubmedclip(cfg, clip_model, train_loader_cache, shot):
    """
    Build or load the cache model for a Hugging Face CLIP-based backbone.
    Arg:
        cfg: configuration dictionary.
        clip_model: Hugging Face CLIP-based model.
        train_loader_cache: data loader used to build the cache.
        shot: number of shots used in the experiment.
    Return:
        cache_keys: cached visual features.
        cache_values: cached labels in one-hot format.
    """
    cache_dir = cfg["cache_dir"]
    keys_path = f"{cache_dir}/keys_{shot}shots.pt"
    values_path = f"{cache_dir}/values_{shot}shots.pt"

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        vision_dtype = next(clip_model.vision_model.parameters()).dtype
        device = next(clip_model.parameters()).device

        with torch.no_grad():
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []

                print(f"Augment Epoch: {augment_idx} / {cfg['augment_epoch']}")
                for images, target in tqdm(train_loader_cache):
                    images = images.to(
                        device=device, dtype=vision_dtype, non_blocking=True
                    )

                    image_features = clip_model.get_image_features(pixel_values=images)
                    train_features.append(image_features)

                    if augment_idx == 0:
                        target = target.to(device=device, non_blocking=True)
                        cache_values.append(target)

                cache_keys.append(
                    torch.cat(train_features, dim=0).unsqueeze(0)
                )

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).contiguous()
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, keys_path)
        torch.save(cache_values, values_path)

    else:
        cache_keys = torch.load(keys_path, map_location="cpu")
        cache_values = torch.load(values_path, map_location="cpu")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    """
    Precompute or load image features and labels for the original CLIP model.
    Arg:
        cfg: configuration dictionary.
        split: dataset split name.
        clip_model: CLIP backbone model.
        loader: data loader for the split.
    Return:
        features: normalized image features.
        labels: corresponding labels.
    """    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not cfg["load_pre_feat"]:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(device), target.to(device)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg["cache_dir"] + "/" + split + "_f.pt")
        torch.save(labels, cfg["cache_dir"] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f.pt")
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l.pt")

    return features, labels


def pre_load_features_openclip(cfg, split, clip_model, loader):
    """
    Precompute or load image features and labels for OpenCLIP-like backbones.
    Arg:
        cfg: configuration dictionary.
        split: dataset split name.
        clip_model: OpenCLIP-like model.
        loader: data loader for the split.
    Return:
        features: normalized image features.
        labels: corresponding labels.
    """
    f_path = os.path.join(cfg["cache_dir"], f"{split}_f.pt")
    l_path = os.path.join(cfg["cache_dir"], f"{split}_l.pt")

    if not cfg["load_pre_feat"]:
        features, labels = [], []
        device = next(clip_model.parameters()).device
        img_dtype = next(clip_model.parameters()).dtype

        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(device=device, dtype=img_dtype, non_blocking=True)
                target = target.to(device=device, non_blocking=True)

                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)
        torch.save(features, f_path)
        torch.save(labels, l_path)

    else:
        features = torch.load(f_path, map_location=device)
        labels = torch.load(l_path, map_location=device)

    return features, labels


def _unwrap_pixel_values(images):
    """
    Extract pixel values from nested Hugging Face processor outputs.
    Arg:
        images: images or processor outputs.
    Return:
        pixel_values: tensor of pixel values.
    """
    if torch.is_tensor(images):
        return images

    if isinstance(images, dict) and "pixel_values" in images:
        pv = images["pixel_values"]
        return _unwrap_pixel_values(pv)

    if hasattr(images, "pixel_values"):
        pv = images.pixel_values
        return _unwrap_pixel_values(pv)

    if hasattr(images, "data") and isinstance(images.data, dict) and "pixel_values" in images.data:
        pv = images.data["pixel_values"]
        return _unwrap_pixel_values(pv)

    if isinstance(images, (list, tuple)):
        pvs = []
        for it in images:
            pv = _unwrap_pixel_values(it)
            pvs.append(pv)

        pvs2 = []
        for pv in pvs:
            if torch.is_tensor(pv) and pv.dim() == 4 and pv.size(0) == 1:
                pv = pv.squeeze(0)
            pvs2.append(pv)

        if all(torch.is_tensor(pv) for pv in pvs2):
            return torch.stack(pvs2, dim=0)

        raise TypeError(f"List elements are not tensors after unwrap. Types: {[type(x) for x in pvs2]}")

    raise TypeError(f"Unsupported images type: {type(images)}")


def pre_load_features_hfclip(cfg, split, clip_model, loader):
    """
    Precompute or load image features and labels for a Hugging Face CLIP-based model.
    Arg:
        cfg: configuration dictionary.
        split: dataset split name.
        clip_model: Hugging Face CLIP-based model.
        loader: data loader for the split.
    Return:
        features: normalized image features.
        labels: corresponding labels.
    """
    f_path = os.path.join(cfg["cache_dir"], f"{split}_f.pt")
    l_path = os.path.join(cfg["cache_dir"], f"{split}_l.pt")

    device = next(clip_model.parameters()).device
    vision_dtype = next(clip_model.vision_model.parameters()).dtype

    if not cfg["load_pre_feat"]:
        features, labels = [], []
        with torch.no_grad():
            for images, target in tqdm(loader):

                pixel_values = _unwrap_pixel_values(images)

                print("RAW pixel_values:", type(pixel_values), getattr(pixel_values, "shape", None))

                if isinstance(pixel_values, (list, tuple)):
                    pixel_values = _unwrap_pixel_values(pixel_values)

                while torch.is_tensor(pixel_values) and pixel_values.dim() > 4 and pixel_values.size(0) == 1:
                    pixel_values = pixel_values.squeeze(0)

                if torch.is_tensor(pixel_values) and pixel_values.dim() == 5 and pixel_values.size(1) == 1:
                    pixel_values = pixel_values.squeeze(1)

                if torch.is_tensor(pixel_values) and pixel_values.dim() == 5 and pixel_values.size(1) > 1:
                    pixel_values = pixel_values[:, 0]

                if torch.is_tensor(pixel_values) and pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)

                print("FINAL pixel_values:", tuple(pixel_values.shape))

                assert pixel_values.dim() == 4, f"Expected 4D (B,3,H,W), got {tuple(pixel_values.shape)}"

                pixel_values = pixel_values.to(device=device, dtype=vision_dtype, non_blocking=True)
                image_features = clip_model.get_image_features(pixel_values=pixel_values)

                target = target.to(device=device, non_blocking=True)

                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)
        torch.save(features, f_path)
        torch.save(labels, l_path)
    else:
        features = torch.load(f_path, map_location=device)
        labels = torch.load(l_path, map_location=device)

    return features, labels


def encode_image_any(model, images, kind):
    """
    Encode images using the appropriate image encoder interface for the selected backbone.
    Arg:
        model: backbone model.
        images: input image batch.
        kind: backbone family identifier.
    Return:
        image_features: encoded image features.
    """
    device = next(model.parameters()).device
    images = images.to(device=device, non_blocking=True)

    if kind == "hf":
        vision_dtype = next(model.vision_model.parameters()).dtype
        return model.get_image_features(pixel_values=images.to(dtype=vision_dtype))

    elif kind in ["openclip", "conch", "biomedclip", "quilt"]:
        img_dtype = next(model.parameters()).dtype
        return model.encode_image(images.to(dtype=img_dtype))

    elif kind == "clip":
        return model.encode_image(images.float())

    else:
        raise ValueError(f"Unknown kind={kind}")


def run_tip_adapter(
    cfg,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
):
    """
    Evaluate Tip-Adapter by searching hyperparameters on the validation set
    and testing the best configuration on the test set.
    Arg:
        cfg: configuration dictionary.
        cache_keys: cached visual features.
        cache_values: cached labels in one-hot format.
        val_features: validation image features.
        val_labels: validation labels.
        test_features: test image features.
        test_labels: test labels.
        clip_weights: text classifier weights.
    Return:
        None
    """

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100.0 * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg["init_beta"], cfg["init_alpha"]

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        cfg, cache_keys, cache_values, val_features, val_labels, clip_weights
    )

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100.0 * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(
    cfg,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    clip_model,
    train_loader_F,
    extra,
):
    """
    Fine-tune the Tip-Adapter cache, perform validation with early stopping,
    search hyperparameters, and evaluate the best adapter on the test set.
    Arg:
        cfg: configuration dictionary.
        cache_keys: cached visual features.
        cache_values: cached labels in one-hot format.
        val_features: validation image features.
        val_labels: validation labels.
        test_features: test image features.
        test_labels: test labels.
        clip_weights: text classifier weights.
        clip_model: backbone model.
        train_loader_F: fine-tuning data loader.
        extra: dictionary containing backend metadata.
    Return:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable the cached keys to be learnable
    D, Ncache = cache_keys.shape
    adapter = nn.Linear(D, Ncache, bias=False).to(device=device, dtype=torch.float32)
    adapter.weight = nn.Parameter(cache_keys.t().contiguous())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], eps=1e-4)
    warmup_scheduler = warmup.LinearWarmup(optimizer, 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg["train_epoch"] * len(train_loader_F)
    )

    beta, alpha = cfg["init_beta"], cfg["init_alpha"]
    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0

    best_ckpt_path = os.path.join(
        cfg["cache_dir"], f"best_F_{cfg['shots']}shots.pt"
    )
    tmp_ckpt_path = os.path.join(
        cfg["cache_dir"], f"best_F_{cfg['shots']}shots.tmp.pt"
    )

    for train_idx in range(cfg["train_epoch"]):
        epoch = train_idx + 1

        # -------------------- Train --------------------
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []

        print(f"Train Epoch: {epoch} / {cfg['train_epoch']}")

        for images, target in tqdm(train_loader_F):
            images, target = images.to(device), target.to(device)

            with torch.no_grad():
                image_features = encode_image_any(clip_model, images, extra["kind"])
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            image_features = image_features.float()
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100.0 * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with warmup_scheduler.dampening():
                scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            "LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}".format(
                current_lr,
                correct_samples / all_samples,
                correct_samples,
                all_samples,
                sum(loss_list) / len(loss_list),
            )
        )

        # -------------------- Eval --------------------
        adapter.eval()
        with torch.no_grad():
            affinity_val = adapter(val_features)
            cache_logits_val = ((-1) * (beta - beta * affinity_val)).exp() @ cache_values
            clip_logits_val = 100.0 * val_features @ clip_weights
            tip_logits_val = clip_logits_val + cache_logits_val * alpha
            val_acc = cls_acc(tip_logits_val, val_labels)

        print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(val_acc))

        if epoch <= 10:
            print(
                f"Early-stopping warmup epoch {epoch}/{10} "
                f"(no stopping, no patience counting)"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

                if os.path.exists(tmp_ckpt_path):
                    os.remove(tmp_ckpt_path)

                torch.save(adapter.weight.detach().cpu(), tmp_ckpt_path)
                os.replace(tmp_ckpt_path, best_ckpt_path)

            epochs_no_improve = 0
            continue

        # -------------------- Early stopping logic --------------------
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            print(
                f"New best validation accuracy: {best_val_acc:.2f} "
                f"at epoch {best_epoch}"
            )

            if os.path.exists(tmp_ckpt_path):
                os.remove(tmp_ckpt_path)

            torch.save(adapter.weight.detach().cpu(), tmp_ckpt_path)
            os.replace(tmp_ckpt_path, best_ckpt_path)

        else:
            epochs_no_improve += 1
            print(
                f"No improvement for {epochs_no_improve} epoch(s). "
                f"Best val accuracy: {best_val_acc:.2f} at epoch {best_epoch}"
            )

            if epochs_no_improve >= 10:
                print(
                    f"Early stopping triggered after {10} epochs "
                    f"without improvement."
                )
                break

    # -------------------- Reload best model --------------------
    if os.path.exists(best_ckpt_path):
        best_weight = torch.load(best_ckpt_path, map_location=device)
        adapter.weight = nn.Parameter(best_weight.to(device=device, dtype=torch.float32))
        print(
            f"**** Reloaded best adapter from epoch {best_epoch} "
            f"with val accuracy {best_val_acc:.2f}. ****\n"
        )
    else:
        print("Warning: no best checkpoint found, using current adapter.")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        clip_weights,
        adapter=adapter,
    )

    print("\n-------- Evaluating on the test set. --------")
    with torch.no_grad():
        affinity_test = adapter(test_features)
        cache_logits_test = ((-1) * (best_beta - best_beta * affinity_test)).exp() @ cache_values.to(
            affinity_test.dtype
        )
        clip_logits_test = 100.0 * test_features @ clip_weights
        tip_logits_test = clip_logits_test + cache_logits_test * best_alpha
        test_acc = cls_acc(tip_logits_test, test_labels)
    print("**** Tip-Adapter-F's final test accuracy: {:.2f}. ****\n".format(test_acc))

    if os.path.exists(best_ckpt_path):
        os.remove(best_ckpt_path)

    if os.path.exists(tmp_ckpt_path):
        os.remove(tmp_ckpt_path)


def main():
    """
    Load the configuration, prepare the model and dataset, build the cache,
    precompute features, and run Tip-Adapter and Tip-Adapter-F.
    Return:
        None
    """
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg["shots"] = args.shots
    dataset_log = cfg["dataset"]
    trainer_log = "Tip-Adapter"
    backbone_log = cfg["backbone"].replace("/", "")
    log_dir = Path(
        f"./Cytology_Benchmark/output/"
        f"{dataset_log}/{trainer_log}/{backbone_log}/{args.shots}shots/seed{args.seed}"
    )

    if log_dir.exists() and run_is_completed(log_dir):
        print(f"[SKIP] already done: {log_dir}")
        return
    if log_dir.exists() and not run_is_completed(log_dir):
        print(f"[RE-RUN] directory exists but run incomplete: {log_dir}")

    setup_logging(cfg, args.shots, args.seed)

    cache_dir = os.path.join(
        "./caches",
        cfg["dataset"],
        backbone_log,
        f"{args.shots}shots",
        f"seed{args.seed}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # ----------------------- Model -----------------------
    clip_model, preprocess, extra = load_backbone_and_preprocess(cfg)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # ----------------------- Prepare dataset -----------------------
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])

    val_loader = build_data_loader(
        data_source=dataset.val,
        batch_size=64,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
    )
    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=64,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
    )

    if cfg["backbone"] == "Conch":
        train_tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=448,
                    scale=(0.5, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        train_tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.5, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    train_loader_cache = build_data_loader(
        data_source=dataset.train_x,
        batch_size=8,
        tfm=train_tranform,
        is_train=True,
        shuffle=False,
    )
    train_loader_F = build_data_loader(
        data_source=dataset.train_x,
        batch_size=8,
        tfm=train_tranform,
        is_train=True,
        shuffle=True,
    )

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    if cfg["backbone"] == "Biomedclip":
        clip_weights = biomedclip_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    elif cfg["backbone"] in ["Quilt-B/16", "Quilt-B/32"]:
        clip_weights = quilt_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    elif cfg["backbone"] == "Conch":
        clip_weights = conch_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        clip_weights = pubmedclip_classifier(
            dataset.classnames, dataset.template, clip_model, extra["tokenizer"]
        )
    else:
        clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    if cfg["backbone"] in ["Biomedclip", "Quilt-B/16", "Quilt-B/32", "Conch"]:
        cache_keys, cache_values = build_cache_biomed(
            cfg, clip_model, train_loader_cache, args.shots
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        cache_keys, cache_values = build_cache_pubmedclip(
            cfg, clip_model, train_loader_cache, args.shots
        )
    else:
        cache_keys, cache_values = build_cache_model(
            cfg, clip_model, train_loader_cache, args.shots
        )

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    if cfg["backbone"] in ["Biomedclip", "Quilt-B/16", "Quilt-B/32", "Conch"]:
        val_features, val_labels = pre_load_features_openclip(
            cfg, "val", clip_model, val_loader
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        val_features, val_labels = pre_load_features_hfclip(
            cfg, "val", clip_model, val_loader
        )
    else:
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg["backbone"] in ["Biomedclip", "Quilt-B/16", "Quilt-B/32", "Conch"]:
        test_features, test_labels = pre_load_features_openclip(
            cfg, "test", clip_model, test_loader
        )
    elif cfg["backbone"] in ["PubMedCLIP-B/32", "PLIP-B/32"]:
        test_features, test_labels = pre_load_features_hfclip(
            cfg, "test", clip_model, test_loader
        )
    else:
        test_features, test_labels = pre_load_features(
            cfg, "test", clip_model, test_loader
        )

    clip_weights = clip_weights.to(device)
    cache_keys = cache_keys.to(device)
    cache_values = cache_values.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    cache_keys = cache_keys.float()
    cache_values = cache_values.float()
    val_features = val_features.float()
    test_features = test_features.float()
    clip_weights = clip_weights.float()

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
    )

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(
        cfg,
        cache_keys,
        cache_values,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
        clip_model,
        train_loader_F,
        extra,
    )


if __name__ == "__main__":
    main()
