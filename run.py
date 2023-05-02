import os
import copy
import pytorch_lightning as pl

from dvp.config import ex
from dvp.modules import DVP_BERT,DVP_T5
from dvp.datamodules.multitask_datamodule import MTDataModule
import torch.distributed as dist
from dvp.datasets import VQAv2Dataset
from dvp.datasets import GQADataset
from dvp.datasets import SNLIVEDataset

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    if _config['search_stage']:
        if _config['datasets'] == ["vqa"]:
            kab_val_dataset = VQAv2Dataset(
                _config["data_root"],
                _config["val_transform_keys"],
                split="val",
                image_size=_config["image_size"],
                tokenizer=_config["tokenizer"],
                max_text_len=_config["max_text_len"],
            )
            print('Using VQA_val Dataset for KAB-APP!')

        if _config['datasets'] == ["gqa"]:
            kab_val_dataset = GQADataset(
                _config["data_root"],
                _config["val_transform_keys"],
                split="val",
                image_size=_config["image_size"],
                tokenizer=_config["tokenizer"],
                max_text_len=_config["max_text_len"],
            )
            print('Using GQA_testdev Dataset for KAB-APP!')

        if _config['datasets'] == ["snli_ve"]:
            kab_val_dataset = SNLIVEDataset(
                _config["data_root"],
                _config["val_transform_keys"],
                split="val",
                image_size=_config["image_size"],
                tokenizer=_config["tokenizer"],
                max_text_len=_config["max_text_len"],
            )
            print('Using SNLI_VE_val Dataset for KAB-APP!')

    else:
        kab_val_dataset = None

    if _config['language_model'] == 'BERT':
        model = DVP_BERT(_config, kab_val_dataset)
    elif _config['language_model'] == 'T5':
        model = DVP_T5(_config, kab_val_dataset)

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
