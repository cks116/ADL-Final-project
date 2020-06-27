import argparse
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    # Testset Path
    parser.add_argument("--testpath")

    # training data
    parser.add_argument("--largeset", action='store_const', default=False, const=True)

    # Model train
    parser.add_argument("--tag", action='store_const', default=False, const=True)
    parser.add_argument("--predict", action='store_const', default=False, const=True)

    args = parser.parse_args()

    return args

args = parse_args()