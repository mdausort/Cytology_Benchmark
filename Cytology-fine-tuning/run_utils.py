import random
import argparse
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_lora_targets(x):
    if x is None:
        return ["q", "k", "v"]

    if isinstance(x, (list, tuple)):
        tokens = []
        for t in x:
            t = str(t).strip().lower()
            t = t.replace("[", "").replace("]", "").replace(",", "")
            if t:
                tokens.append(t)
        return tokens

    s = str(x).strip().lower()
    s = s.replace("[", "").replace("]", "")
    s = s.replace(",", " ")
    tokens = [t.strip() for t in s.split() if t.strip()]
    return tokens


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Seed number")
    parser.add_argument("--log_path", type=str, default=None, help="Path to save logs")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./Cytology_Benchmark/",
        help="Path of your root directory.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/path/to/datasets",
        help="Path of the dataset directory.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./Cytology_Benchmark/output/",
        help="Path of the results directory.",
    )
    parser.add_argument(
        "--features_path", type=str, help="Path of the dataset directory."
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Task name",
        choices=["lora-vlm", "lora-vision", "feature_extract"],
    )
    parser.add_argument("--shots", type=int, default=16, help="Shot number")
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.0,
        help="Percentage of the dataset considered. Used for the third experiment.",
    )

    parser.add_argument(
        "--dataset", type=str, default="mlcc", help="Name of the dataset used"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes considered for the classification task",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="level_1",
        help="This is the level of the hierarchical tree to capture different fine-grained subtype information. Only applicable in the case of hicervix.",
        choices=["level_1", "level_2", "level_3", "class_name"],
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="clip",
        help="Name of the model used",
        choices=[
            "clip-b16",
            "clip-b32",
            "clip-l14",
            "quilt-b16",
            "quilt-b32",
            "pubmedclip",
            "plip",
            "biomedclip",
            "conch",
            "dinobloom-s",
            "dinobloom-b",
            "dinobloom-l",
            "dinobloom-g",
            "uni",
            "vit_google-b16",
            "vit_google-b32",
        ],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-B/16",
        help="Configuration of the model's backbone",
        choices=[
            "ViT-L/14",
            "ViT-L/16",
            "ViT-B/16",
            "ViT-B/32",
            "ViT-B/14",
            "ViT-S/14",
            "ViT-G/14",
        ],
    )

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Size of the batch")

    parser.add_argument(
        "--position",
        type=str,
        default="all",
        help="where to put the LoRA modules",
        choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="both",
        choices=["text", "vision", "both"],
        help="It is the part of the model on which we want apply LoRA, either on the visual or textual part.",
    )
    parser.add_argument(
        "--params",
        type=parse_lora_targets,
        default="q,k,v,o",
        help="list of attention matrices where putting a LoRA (e.g. q,k,v or [q,k,v])",
    )
    parser.add_argument(
        "--r", type=int, default=2, help="the rank of the low-rank matrices"
    )

    parser.add_argument("--alpha", default=1, type=int, help="scaling (see LoRA paper)")
    parser.add_argument(
        "--dropout_rate",
        default=0.25,
        type=float,
        help="dropout rate applied before the LoRA module",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="path to save the lora modules after training, not saved if None",
    )
    parser.add_argument(
        "--filename",
        default="lora_weights",
        help="file name to save the lora weights (.pt extension will be added)",
    )
    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="only evaluate the LoRA modules (save_path should not be None)",
    )

    args = parser.parse_args()
    return args
