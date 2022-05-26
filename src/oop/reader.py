# Contains everything to read input from different sources
import os
from pathlib import Path
import argparse


class Reader:
    def __init__(self):
        self.cwd = os.getcwd()

    def get_input(self):
        parser = argparse.ArgumentParser()
        # Required positional arguments
        parser.add_argument('act_energy', type=float,
                            help='A required float activation energy argument')
        parser.add_argument('bump_amp', type=float,
                            help='A required float amplitude of a halfwave argument')
        parser.add_argument('bump_wn', type=float,
                            help='A required float wavenumber of halfwave argument')
        # parser.add_argument('dom_length', type=float,
        #                     help='A required float domain length argument')
        # parser.add_argument('num_nodes', type=int,
        #                     help='A required int number of nodes argument')
        args = parser.parse_args()
        return vars(args)


if __name__=='__main__':
    reader = Reader()
    input = reader.get_input()
    print(input)
