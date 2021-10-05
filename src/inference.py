import pandas as pd
import numpy as np

import torch

from dataset import PetfinderDataModule
from model import Model
from train import make_parse



test_df = pd.read_csv()
test_dataset = PetfinderDataModule()

preds = []

for fold in range("flod数"):
    # model呼び出し
    model = Model(config)

    # ハイパラ呼び出し
    model.load_state_dict(torch.load("パラメータのパス"))
    model = model.cuda().eval()

    # TODO: test用のデータセット必要？

    test_predictions = model.predict(test_dataset, batch_size=...)
    preds.append(test_predictions.ravel().tolist())



df_test["Pawpularity"] = np.mean(preds, axis=1) 
df_test = df_test[["Id", "Pawpularity"]]
df_test.to_csv("submission.csv", index=False)





