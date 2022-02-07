import numpy as np
import pandas as pd
import read_csv

T = 100
GENERATOR_COST = -10

class State:

    def __init__(self, SoC, is_gen_on, dem, gen, dem_pred, gen_pred):
        self.battery_charge = SoC
        self.is_generator_on = is_gen_on
        self.demand = dem
        self.generation = gen
        self.demand_prediction = dem_pred
        self.gen_prediction = gen_pred

    # array = numpy.ndarray((T,),dtype=numpy.object)
    # for i in range(10):
    #     array[i] = State()
    #
    # for i in range(10):
    #     print array[i].method()


def main():
    data_dict = read_csv.read_files()

    for t in range(T):
        pass


def r(s: State, a):
    # if generator is on or we are turning it on
    if s.is_generator_on or (not s.is_generator_on and a == 1):
        return GENERATOR_COST
    else:
        return 0


def f(s: State, a):
    pass




if __name__ == '__main__':
    main()

