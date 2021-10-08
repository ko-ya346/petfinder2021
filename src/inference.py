import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import torch
from dataset import PetfinderDataModule, PetfinderDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train import make_parse
from utils.factory import read_yaml
from utils.transform import get_default_transforms

warnings.filterwarnings("ignore")


def main():
    args = make_parse().parse_args()
    conf = read_yaml(fpath=args.config)

    if args.knotebook:
        conf.dataset.test_df = "../input/petfinder-pawpularity-score"
        conf.dataset.test_img_dir = "../input/petfinder-pawpularity-score/test"
        ex_output_dir = "../input/kamarinskaya"
    else:
        ex_output_dir = os.path.join(conf.General.output_dir, conf.General.name)

    test_df = pd.read_csv(os.path.join(conf.dataset.test_df, "test.csv"))
    print(test_df.shape)

    test_df["Id"] = test_df["Id"].apply(
        lambda x: os.path.join(conf.dataset.test_img_dir, x + ".jpg")
    )
    test_dataset = PetfinderDataset(
        test_df,
        conf.dataset.img_height,
        conf.dataset.img_width,
    )

    test_dataloader = DataLoader(test_dataset, **conf.dataloader.test)

    preds = []

    for ckpt_path in list(glob(f"{ex_output_dir}/*.ckpt")):
        tmp = []
        model = Model(conf)

        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        model = model.cuda().eval()

        with torch.no_grad():
            for image in tqdm(test_dataloader):
                image = get_default_transforms()["test"](image)
                image = image.to("cuda")
                test_predictions = (
                    model.forward(image).squeeze(1).sigmoid().detach().cpu() * 100.0
                )
                tmp.extend(test_predictions.flatten().tolist())

        preds.append(tmp)

    test_df["Pawpularity"] = np.mean(preds, axis=0)
    test_df["Id"] = test_df["Id"].apply(lambda x: x.split("/")[-1])
    test_df["Id"] = test_df["Id"].apply(lambda x: x.split(".")[0])
    test_df = test_df[["Id", "Pawpularity"]]
    test_df.to_csv("submission.csv", index=False)
    print(test_df.head())


if __name__ == "__main__":
    main()
