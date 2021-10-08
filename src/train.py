import argparse
import os
import warnings

import pandas as pd
import torch
from dataset import PetfinderDataModule
from model import Model
from sklearn.model_selection import StratifiedKFold
from utils.factory import get_transform, read_yaml

warnings.filterwarnings("ignore")


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--knotebook", action="store_true", help="run kaggle notebook")
    arg("--config", default=None, type=str, help="config path")
    return parser


def main():
    import pytorch_lightning as pl
    from pytorch_lightning import callbacks
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.utilities.seed import seed_everything

    args = make_parse().parse_args()
    conf = read_yaml(fpath=args.config)

    if args.debug:
        conf.General.epoch = 3

    conf_aug = conf.Augmentation["train"]
    transform = get_transform(conf_aug)

    # BackwardテンソルのNaNチェックを行う
    torch.autograd.set_detect_anomaly(True)
    seed_everything(conf.General.seed)

    # df読み込む
    df = pd.read_csv(os.path.join(conf.dataset.train_df, "train.csv"))
    df["Id"] = df["Id"].apply(
        lambda x: os.path.join(conf.dataset.train_img_dir, x + ".jpg")
    )

    skf = StratifiedKFold(
        n_splits=conf.dataset.kfold, shuffle=True, random_state=conf.General.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        train_df = df.loc[train_idx].reset_index(drop=True)

        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderDataModule(train_df, val_df, conf)
        model = Model(conf)
        earlystopping = EarlyStopping(monitor="valid_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="valid_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            dirpath=os.path.join(conf.General.output_dir, conf.General.name),
        )

        logger = TensorBoardLogger(
            os.path.join(conf.General.output_dir, conf.General.name)
        )
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=conf.General.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earlystopping],
            **conf.General.trainer,
        )

        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
