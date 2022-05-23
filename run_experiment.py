"""
- TODO -
- TODO create its own conda environment
- TODO -
- TODO *** polish names and everything
- TODO -
"""
from params import *
from src.generate_data import generate_data
from src.generate_plots import generate_plots


def main():

    print("Begin experiment\n")

    generate_data(params)
    generate_plots(params)

    print("Finish experiment")

if __name__ == '__main__':
    
    main()