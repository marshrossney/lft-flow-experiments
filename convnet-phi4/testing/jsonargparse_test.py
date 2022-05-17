from jsonargparse import ArgumentParser, ActionConfigFile
from dataclasses import dataclass


@dataclass
class Test:
    dog: int = 10
    cat: str = "hello mr cat"
    rat: float = 69.69


parser = ArgumentParser()
parser.add_argument("--test", type=Test, default=Test())
parser.add_argument("-c", "--config", action=ActionConfigFile)

config = parser.parse_args()

print(config)
