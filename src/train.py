import argparse

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

if __name__ == "__main__":
    main()
