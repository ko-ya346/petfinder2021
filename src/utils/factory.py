import torchvision.transforms as T
import yaml
from addict import Dict

# import albumentations as alb


def get_transform(conf_alb):
    def get_object(trans):
        if trans.name in {"Compose", "OneOf"}:

            alb_tmp = [get_object(aug) for aug in trans.member]
            print(alb_tmp)
            return getattr(T, trans.name)(alb_tmp, **trans.params)
        if hasattr(T, trans.name):
            return getattr(T, trans.name)(**trans.params)

    print(conf_alb)
    if conf_alb is None:
        albs = list()
    else:
        albs = [get_object(aug) for aug in conf_alb]
    print(albs)
    return T.Compose(albs)


def read_yaml(fpath: str) -> dict:
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
