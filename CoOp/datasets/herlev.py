import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


NEW_CNAMES = {
    "carcinoma_in_situ": "Carcinoma in situ",
    "light_dysplastic": "Light dysplastic",
    "moderate_dysplastic": "Moderate dysplastic",
    "normal_columnar": "Normal columnar",
    "normal_intermediate": "Normal intermediate",
    "normal_superficiel": "Normal Superficiel",
    "severe_dysplastic": "Severe dysplastic"
}

SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
}


@DATASET_REGISTRY.register()
class Herlev(DatasetBase):
    dataset_dir = "herlev"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.splits_dir = os.path.join(self.dataset_dir, "splits")

        # few-shot cache (optionnel)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        os.makedirs(self.split_fewshot_dir, exist_ok=True)

        train = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["train"]))
        val = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["val"]))
        test = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["test"]))

        # few-shot (même logique que EuroSAT)
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=16)
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def read_txt_split(self, filepath):
        """
        Each line:  im_Dyskeratotic/073_02.bmp 0
        Path is relative to images/ (i.e., images/<that_path>)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Split file not found: {filepath}")

        items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # robust split: allow spaces/tabs
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad line format in {filepath} at line {ln}: {line}")

                rel_impath = parts[0]
                label = int(parts[1])

                # classname from folder name (first component)
                class_folder = rel_impath.split("/")[0]
                classname = NEW_CNAMES.get(class_folder, class_folder)

                impath = os.path.join(self.image_dir, rel_impath)

                # optionnel: check existence (tu peux enlever si ça ralentit)
                if not os.path.exists(impath):
                    raise FileNotFoundError(
                        f"Image not found (from split): {impath} (line {ln} in {filepath})"
                    )

                items.append(Datum(impath=impath, label=label, classname=classname))

        print(f"Loaded {len(items)} items from {filepath}")
        return items
