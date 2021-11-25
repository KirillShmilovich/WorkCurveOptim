import yaml
import sys

from src.dimer import DIMER
from src.anm import ANM
from src.optim import Optimizer


def main(args):
    dimer = DIMER(**args)
    anm = ANM(**args)
    optim = Optimizer(dimer, anm, **args)
    optim.optimize()


if __name__ == "__main__":
    hparams_file = sys.argv[1]

    with open(hparams_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    main(args)
