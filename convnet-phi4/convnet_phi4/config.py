from __future__ import annotations


from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import PositiveInt


parser = ArgumentParser()
parser.add_argument("--lattice-shape", type=list[PositiveInt])

parser.add_argument("--n-train", type=PositiveInt)
parser.add_argument("--n-batch", type=PositiveInt)
parser.add_argument("--n-batch-val", type=PositiveInt)
parser.add_argument("--val-interval", type=PositiveInt)

parser.add_argument("--flow", type=int)  # TODO
parser.add_argument("--target", type=int)  # TODO

parser.add_argument("-c", "--config", action=ActionConfigFile)
