import os

from .utils import Datum, DatasetBase


template = ["a photo of a {}."]

NEW_CNAMES = {
    "ACTINO": "Bacteria morphologically consistent with Candida spp.",
    "ADC": "Adenocarcinoma",
    "AGC-FN": "Atypical glandular cell- favor neoplastic",
    "AGC-NOS": "Atypical glandular cell- not otherwise specified",
    "ASC-H": "Atypical squamous cells cannot exclude high grade squamous intraepithelial lesion",
    "ASC-US": "Atypical squamous cells of undetermined significance",
    "Atrophy": "Atrophy",
    "CC": "Coccobacilli/shift in flora suggestive of bacterial vaginosis",
    "ECC": "Endocervical cell",
    "EMC": "Endometrial cell",
    "FUNGI": "Fungal organisms morphologically consistent with Actinomyces spp.",
    "HCG": "Hyperchromatic crowded groups",
    "HSIL": "High-grade squamous intraepithelial lesion",
    "HSV": "Cellular changes consistent with herpes simplex virus",
    "LSIL": "Low-grade squamous intraepithelial lesion",
    "MPC": "Metaplasia cell",
    "Normal": "Normal cell",
    "PG": "Pseudokoilocytes by glycogen",
    "RPC": "Repair cell",
    "SCC": "Squamous cell carcinoma",
    "TRI": "Trichomonas vaginalis"
}

SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
}


class HiCervix(DatasetBase):
    dataset_dir = "hicervix"

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.splits_dir = os.path.join(self.dataset_dir, "splits")

        self.template = template

        train = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["train"]))
        val = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["val"]))
        test = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["test"]))

        # few-shot “simple” (comme ton EuroSAT simple)
        n_shots_val = 16
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

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
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad line in {filepath} (line {ln}): {line}")

                rel_path = parts[0]
                label = int(parts[1])

                class_folder = rel_path.split("/")[0]
                classname = NEW_CNAMES.get(class_folder, class_folder)

                impath = os.path.join(self.image_dir, rel_path)
                if not os.path.exists(impath):
                    raise FileNotFoundError(
                        f"Image not found: {impath} (from {filepath}:{ln})"
                    )

                items.append(Datum(impath=impath, label=label, classname=classname))

        print(f"Loaded {len(items)} items from {filepath}")
        return items
