def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float=1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(
                self.cfg.Model.name, pretrained=True, num_classes=0, in_chans=3)
            )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(num_features, self.cfg.Model.out_channel
            )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        _, pred, labels = self.__share_step(batch, "valid")
        return {"pred": pred, "labels": labels}

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        images = get_transform(self.cfg.Augmentation[mode])
        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeese(1)
            loss = (
                self._criterion(logits, target_a) * lam
                + self._criterion(logits, target_b) * (1 - lam)
                )
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)

        pred = logits.sigmoid().detach().cpu() * 100.
        labels = labels.detach().spu() * 100.
        return loss, pred, labels
    
    def training_epoch_end(self, outputs):
        
