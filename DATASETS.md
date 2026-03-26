# How to install datasets

We suggest putting all datasets under the same folder (e.g., `$DATA`) to ease management and avoid modifying the source code.

```
$DATA/
|–– apacc/
|–– bcfc/
|–– bloodmnist/
|–– ...
```

If you already have datasets stored elsewhere, you can create symbolic links in `$DATA/dataset_name` to avoid duplication.

## Expected structure

All datasets should follow the same structure:

```
$DATA/
|–– dataset_name/
|   |–– images/
|   |–– splits/
|   |–– classnames.txt
```

* `images/`: images organized by class (subfolders)
* `splits/`: contains `train.txt`, `val.txt`, `test.txt`
* `classnames.txt`: list of class names

## Splits

To ensure reproducibility and fair comparison, we use **fixed train/val/test splits**:

* When available, we use **official splits**
* Otherwise, splits are generated using our preprocessing script

## Datasets

* [APACC](#apacc)
* [BCFC](#bcfc)
* [BloodMNIST](#bloodmnist)
* [BMCD](#bmcd)
* [BMT](#bmt)
* [FNAC](#fnac)
* [Herlev](#herlev)
* [HiCervix](#hicervix)
* [MLCC](#mlcc)
* [SiPaKMed](#sipakmed)

## General preparation procedure

For most datasets:

1. Download and extract the dataset
2. Organize images into class-specific folders (if not already done)
3. Run our preprocessing script to generate `train/val/test` splits

## Dataset-specific instructions

### APACC

* Create a folder named `apacc/` under `$DATA`
* Download the dataset
* Organize images by class (e.g., `benign/`, `malignant/`)

### BCFC

* Create `bcfc/` under `$DATA`
* Download and extract the dataset
* Ensure images are grouped by class

### BloodMNIST

* Create `bloodmnist/` under `$DATA`
* Download the dataset from MedMNIST
* Convert provided splits to our format

### BMCD

* Create `bmcd/` under `$DATA`
* Download and extract the dataset

### BMT

* Create `bmt/` under `$DATA`
* Download and extract the dataset

## ### FNAC

* Create `fnac/` under `$DATA`
* Download and extract the dataset

### Herlev

* Create `herlev/` under `$DATA`
* Download the dataset

### HiCervix

* Create `hicervix/` under `$DATA`
* Download and extract the dataset

### MLCC

* Create `mlcc/` under `$DATA`
* Download and extract the dataset

### SiPaKMed

* Create `sipakmed/` under `$DATA`
* Download the dataset
