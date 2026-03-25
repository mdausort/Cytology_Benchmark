import os
from run_utils import set_random_seed, get_arguments


if __name__ == "__main__":

    args = get_arguments()
    set_random_seed(args.seed)

    features_path = os.path.join(args.root_path, "features")

    class_counts = {
        "apacc": 4,
        "bcfc": 2,
        "bloodmnist": 8,
        "bmcd": 21,
        "bmt": 3,
        "fnac2019": 2,
        "herlev": 7,
        "hicervix": 21,
        "mlcc": 4,
        "sipakmed": 5,
    }

    if args.dataset in [
        "apacc",
        "mlcc",
        "bcfc",
        "bloodmnist",
        "bmcd",
        "bmt",
        "fnac2019",
        "herlev",
        "hicervix",
        "sipakmed",
    ]:
        num_classes = class_counts[args.dataset]

    else:
        raise RuntimeError(
            "The code is not set for this dataset. Modify the code to take it into account."
        )

    os.system(
        f"python3 main.py \
        --seed {args.seed} \
        --root_path {args.root_path} \
        --dataset_path {args.dataset_path} \
        --results_path {args.results_path} \
        --features_path {features_path} \
        \
        --task {args.task} \
        --shots {args.shots} \
        --percentage {args.percentage} \
        \
        --dataset {args.dataset} \
        --num_classes {num_classes} \
        --level {args.level} \
        \
        --model_name {args.model_name} \
        --backbone {args.backbone} \
        \
        --lr {args.lr} \
        --n_epochs {args.n_epochs} \
        --batch_size 8 \
        \
        --position all \
        --encoder {args.encoder} \
        --params [q,k,v,o] \
        --r {args.r} "
    )
