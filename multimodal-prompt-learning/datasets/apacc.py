import os
import math
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import write_json, listdir_nohidden


NEW_CNAMES = {
    "bothcells": "Both cells",
    "healthy": "Healthy",
    "rubbish": "Rubbish",
    "unhealthy": "Unhealthy"
}


SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
}


@DATASET_REGISTRY.register()
class APACC(DatasetBase):
    dataset_dir = "apacc"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.splits_dir = os.path.join(self.dataset_dir, "splits")

        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        os.makedirs(self.split_fewshot_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.splits_dir, SPLIT_FILES["train"])):
            train = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["train"]))
            val = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["val"]))
            test = self.read_txt_split(os.path.join(self.splits_dir, SPLIT_FILES["test"]))
        else:
            train, val, test = self.read_and_split_data(
                self.image_dir, new_cnames=NEW_CNAMES
            )
            self.save_split(train, val, test, self.split_path, self.image_dir)

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

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

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
                    raise ValueError(f"Bad line format in {filepath} at line {ln}: {line}")

                rel_impath = parts[0]
                label = int(parts[1])

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

    @staticmethod
    def read_and_split_data(image_dir,
                            p_trn=0.5,
                            p_val=0.2,
                            ignored=[],
                            new_cnames=None):

        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(
            f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test"
        )

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y,
                             classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(
                _collate(images[n_train:n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        assert subsample in ["all", "base", "new"]
        if subsample == "all":
            return args

        dataset = args[0]
        labels = sorted(set(item.label for item in dataset))
        n = len(labels)
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        selected = labels[:m] if subsample == "base" else labels[m:]
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for ds in args:
            ds_new = []
            for item in ds:
                if item.label not in selected:
                    continue
                ds_new.append(
                    Datum(
                        impath=item.impath,
                        label=relabeler[item.label],
                        classname=item.classname,
                    )
                )
            output.append(ds_new)
        return output
