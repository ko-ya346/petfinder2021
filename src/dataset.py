import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class PetfinderDataset(Dataset):
    def __init__(self, df, img_height=256, img_width=256):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        self._transform = T.Resize([img_height, img_width])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self.X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self.y[idx]
            return image, label
        return image


class PetfinderDataModule(LightningDataModule):
    def __init__(self, train_df, val_df, conf):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._conf = conf

    def __create_dataset(self, train=True):
        return (
            PetfinderDataset(
                self._train_df,
                self._conf.dataset.img_height,
                self._conf.dataset.img_width,
            )
            if train
            else PetfinderDataset(
                self._val_df,
                self._conf.dataset.img_height,
                self._conf.dataset.img_width,
            )
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._conf.dataloader.train)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._conf.dataloader.valid)
