import numpy as np
import pandas as pd

T = 100

class State:

    def __init__(self, SoC, dem, pvgen, dem_pred, pvgen_pred):
        self.battery_charge = SoC
        self.demand = dem
        self.pv_gen = pvgen
        self.demand_prediction = dem_pred
        self.pv_gen_prediction = pvgen_pred

    # array = numpy.ndarray((T,),dtype=numpy.object)
    # for i in range(10):
    #     array[i] = State()
    #
    # for i in range(10):
    #     print array[i].method()

def main():
    states =


if __name__ == '__main__':
    main()

