from .vqav2_datamodule import VQAv2DataModule
from .gqa_datamodule import GQADataModule
from .snli_ve_datamodule import SNLIVEDataModule

_datamodules = {
    "vqa": VQAv2DataModule,
    'gqa': GQADataModule,
    'snli_ve': SNLIVEDataModule,
}
