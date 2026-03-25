import clip
import torch
import torch.nn as nn
import open_clip
from datasets import build_dataset
import torchvision.transforms as transforms
from datasets.utils import build_data_loader
from lora import run_lora_text, run_lora, run_lora_features_extractor
from utils import setup_logging
from run_utils import set_random_seed, get_arguments
from transformers import (
    CLIPModel,
    CLIPProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
)
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import conch.open_clip_custom


def load_model_and_preprocess(model_name):
    """
    Load the selected model together with its preprocessing pipeline.
    Arg:
        model_name: name of the model to load.
    Return:
        model: loaded model.
        preprocess: preprocessing function or transform associated with the model.
        hf_processor: Hugging Face processor when required, otherwise None.
    """
    hf_processor = None

    if model_name == "clip-b16":
        model, preprocess = clip.load("ViT-B/16")

    elif model_name == "clip-b32":
        model, preprocess = clip.load("ViT-B/32")

    elif model_name == "clip-l14":
        model, preprocess = clip.load("ViT-L/14")

    elif model_name == "biomedclip":
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    elif model_name == "quilt-b16":
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:wisdomik/QuiltNet-B-16"
        )
    elif model_name == "quilt-b32":
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:wisdomik/QuiltNet-B-32"
        )
    elif model_name == "pubmedclip":
        model = CLIPModel.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )
        hf_processor = CLIPProcessor.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )

        def preprocess(img):
            x = hf_processor(images=img, return_tensors="pt")[
                "pixel_values"
            ]
            return x.squeeze(0)

    elif model_name == "plip":
        model = CLIPModel.from_pretrained("vinid/plip")
        hf_processor = CLIPProcessor.from_pretrained("vinid/plip")

        def preprocess(img):
            x = hf_processor(images=img, return_tensors="pt")["pixel_values"]
            return x.squeeze(0)

    elif model_name == "conch":
        model, preprocess = conch.open_clip_custom.create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
        )

    elif "dinobloom" in model_name:
        if model_name == "dinobloom-s":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_s.bin"
            embed_dim = 384
        elif model_name == "dinobloom-b":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_b.bin"
            embed_dim = 768
        elif model_name == "dinobloom-l":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            path = "/gpfs/home/acad/ucl-elen/mdausort/.cache/huggingface/hub/models--MarrLab--DinoBloom/snapshots/e025b6824330fc57b3b9dfe1f66ec5141c1bc4ff/pytorch_model_l.bin"
            embed_dim = 1024
        elif model_name == "dinobloom-g":
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

    elif model_name == "uni":

        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        preprocess = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

    elif model_name == "vit_google-b16":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        vit_proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        def preprocess(img):
            x = vit_proc(images=img, return_tensors="pt")["pixel_values"]
            return x.squeeze(0)

    elif model_name == "vit_google-b32":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        vit_proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        def preprocess(img):
            x = vit_proc(images=img, return_tensors="pt")["pixel_values"]
            return x.squeeze(0)

    else:
        raise RuntimeError(f"Unknown model name: {model_name}")

    return model, preprocess, hf_processor


def main():
    """
    Parse arguments, load the model, prepare the dataset and data loaders, and launch the selected training or feature extraction routine.
    Return:
        None
    """
    args = get_arguments()
    set_random_seed(args.seed)

    setup_logging(args)
    model, preprocess, hf_processor = load_model_and_preprocess(args.model_name)

    if args.model_name in ["pubmedclip", "plip"]:
        args._hf_processor = hf_processor

    model.eval()

    # Prepare dataset
    print("Preparing dataset.")

    dataset = build_dataset(args.dataset, args.dataset_path, args.shots)

    val_loader = build_data_loader(
        data_source=dataset.val,
        batch_size=256,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
        num_workers=4,
    )
    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=256,
        is_train=False,
        tfm=preprocess,
        shuffle=False,
        num_workers=4,
    )

    train_tranform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.08, 1),
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

    train_loader = build_data_loader(
        data_source=dataset.train_x,
        batch_size=args.batch_size,
        tfm=train_tranform,
        is_train=True,
        shuffle=True,
        num_workers=4,
    )

    if args.task == "lora-vision":
        run_lora(args, model, train_loader, val_loader, test_loader)
    elif args.task == "feature_extract":
        run_lora_features_extractor(args, model, train_loader, val_loader, test_loader)
    else:
        run_lora_text(
            args, model, dataset, train_loader, val_loader, test_loader
        )


if __name__ == "__main__":
    main()
