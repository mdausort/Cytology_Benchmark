from .eurosat import EuroSAT
from .sipakmed import SiPakMed
from .bcfc import BCFC
from .apacc import APACC
from .bloodmnist import BloodMNIST
from .bmcd import BMCD
from .bmt import BMT
from .fnac2019 import FNAC2019
from .herlev import Herlev
from .hicervix import HiCervix
from .mlcc import MLCC


dataset_list = {
    "apacc": APACC,
    "bcfc": BCFC,
    "bloodmnist": BloodMNIST,
    "bmcd": BMCD,
    "bmt": BMT,
    "eurosat": EuroSAT,
    "fnac2019": FNAC2019,
    "herlev": Herlev,
    "hicervix": HiCervix,
    "mlcc": MLCC,
    "sipakmed": SiPakMed
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
