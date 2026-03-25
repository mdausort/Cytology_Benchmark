import os

from .utils import Datum, DatasetBase


template = ["a photo of a {}."]

NEW_CNAMES = {
    "HSIL": "High-Grade Squamous Intraepithelial Lesion",
    "LSIL": "Low-Grade Squamous Intraepithelial Lesion",
    "NIL": "Negative for Intraepithelial Lesion of Malignancy"
}

SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
}


class BMT(DatasetBase):
    dataset_dir = "bmt"

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.splits_dir = os.path.join(self.dataset_dir, "splits")

        self.template = template

        train = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["train"]))
        val = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["val"]))
        test = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["test"]))

        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset(val, num_shots=16)

        super().__init__(train_x=train, val=val, test=test)

    def read_txt_split(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Split file not found: {filepath}")

        items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    rel_impath, label_str = line.rsplit(maxsplit=1)
                    label = int(label_str)
                except ValueError as e:
                    raise ValueError(
                        f"Bad line format in {filepath} at line {ln}: {line!r}"
                    ) from e

                class_folder = rel_impath.split("/")[0]
                classname = NEW_CNAMES.get(class_folder, class_folder)

                impath = os.path.join(self.image_dir, rel_impath)

                if not os.path.exists(impath):
                    raise FileNotFoundError(
                        f"Image not found (from split): {impath} (line {ln} in {filepath})"
                    )

                items.append(Datum(impath=impath, label=label, classname=classname))

        print(f"Loaded {len(items)} items from {filepath}")
        return items
