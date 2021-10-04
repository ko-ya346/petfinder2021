import argparse

from dataset import PetfinderDataModule
from utils.factory import get_transform, read_yaml


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--config", default=None, type=str, help="config path")
    return parser


def main():
    args = make_parse().parse_args()
    conf = read_yaml(fpath=args.config)
    print(conf.keys())
    print(type(conf))

    conf_aug = conf.Augmentation["train"]
    transform = get_transform(conf_aug)

    # df読み込む

    skf = StratifiedKFold(
        n_splits=conf.dataset.kfold, shuffle=True, random_state=conf.General.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderDataModule(train_df, val_df, conf)
        model = Model(conf)
        earlystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )

        logger = TensorBoardLogger(conf.Model.name)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=conf.General.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earlystopping],
            **conf.General.trainer,
        )

        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
