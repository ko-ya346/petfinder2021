import torchvision.transforms as T
import yaml
from addict import Dict


def get_transform(conf_alb):
    def get_object(trans):
        if trans.name in {"Compose", "OneOf"}:

            alb_tmp = [get_object(aug) for aug in trans.member]
            return getattr(T, trans.name)(alb_tmp, **trans.params)
        if hasattr(T, trans.name):
            return getattr(T, trans.name)(**trans.params)

    if conf_alb is None:
        albs = list()
    else:
        albs = [get_object(aug) for aug in conf_alb]
    return T.Compose(albs)


def read_yaml(fpath: str) -> dict:
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
