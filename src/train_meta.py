import argparse
import os
import warnings
from glob import glob

import pandas as pd
import torch
from dataset import PetfinderDataModule
from model import Model_meta
from sklearn.model_selection import StratifiedKFold
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils.factory import get_transform, read_yaml

warnings.filterwarnings("ignore")


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--knotebook", action="store_true", help="run kaggle notebook")
    arg("--config", default=None, type=str, help="config path")
    arg("--output", default=None, type=str, help="logger & ckpt output path")
    arg("--seed", default=20, type=int, help="random seed")
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
    if args.knotebook:
        conf.dataset.train_df = "../input/petfinder-pawpularity-score"
        conf.dataset.train_img_dir = "../input/petfinder-pawpularity-score/train"

    fcols = [
        "Subject Focus",
            "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
        ]

    # output pathを指定しない場合はGeneral.nameというフォルダに保存
    # random seedを変えるだけの学習をするときに、わざわざ.yamlを用意するのが面倒なので
    # 保存先を指定できるようにしてみた
    # 動作未確認
    if args.output is None:
        args.output = conf.General.name

    conf_aug = conf.Augmentation["train"]
    transform = get_transform(conf_aug)

    # BackwardテンソルのNaNチェックを行う
    torch.autograd.set_detect_anomaly(True)
    seed_everything(args.seed)

    # df読み込む
    df = pd.read_csv(os.path.join(conf.dataset.train_df, "train.csv"))
    df["Id"] = df["Id"].apply(
        lambda x: os.path.join(conf.dataset.train_img_dir, x + ".jpg")
    )
    if args.debug:
        df = df.loc[:100]

    skf = StratifiedKFold(
        n_splits=conf.dataset.kfold, shuffle=True, random_state=args.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        train_df = df.loc[train_idx].reset_index(drop=True)

        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderDataModule(train_df, val_df, conf, fcols)
        model = Model_meta(conf, pretrained=True)
        earlystopping = EarlyStopping(monitor="valid_loss", patience=3)
        lr_monitor = callbacks.LearningRateMonitor()
        dirpath = os.path.join(conf.General.output_dir, args.output)
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="valid_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            dirpath=dirpath,
        )

        logger = TensorBoardLogger(save_dir=dirpath)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=conf.General.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earlystopping],
            **conf.General.trainer,
        )

        trainer.fit(model, datamodule=datamodule)

    # CV score 動作未確認
    ex_dir = args.config.split("/")[-1].split(".")[0]
    print(ex_dir)
    path = glob(f"./output/{args.output}/default/version_*/events*")
    print(path)
    idx = 0
    print(path[idx])
    event_acc = EventAccumulator(path[idx], size_guidance={"scalars": 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    print("best_val_loss", min(scalars["valid_loss"]))


if __name__ == "__main__":
    main()
