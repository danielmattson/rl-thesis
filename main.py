import random

import numpy as np
import pandas as pd
from typing import Dict

import read_csv

T = 0
GENERATOR_COST = -10
AVG_GENERATOR_OUTPUT = 13000

# using data from 2017 SEI report, page 71. ECB battery max capacity is 700 Ah
# in more recent CSVs, voltage is ~50 V
# Wh = V * Ah = 50 * 700 = 35000 Wh = 35 kWh
TOTAL_BATTERY_CAPACITY = 35000


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
    data_dict: Dict[str, pd.DataFrame] = read_csv.read_files()
    global T
    T = len(data_dict['demand'])
    global sim_states
    sim_states = create_sim_state_list(data_dict)
    M = 10

    for m in range(M):
        total_reward = 0
        states = [State(100, False, 0, 0, None, None)]
        for t in range(T - 1):
            action = random.choice([True, False])
            states.append(f(states[t], action, t))
            total_reward += r(states[t], action)
        total_reward += r(states[-1], None)

        print(f'reward from random policy: {total_reward}')


def r(s: State, a):
    # if generator is left on or we are turning it on
    if s.is_generator_on:
        return GENERATOR_COST
    else:
        # no cost to rely on renewable energy
        return 0


def f(s: State, a, t) -> State:
    next_sim_state = sim_states[t + 1]
    next_demand = next_sim_state.demand
    next_gen = next_sim_state.generation

    percent_generation = (next_gen + AVG_GENERATOR_OUTPUT if s.is_generator_on else 0) / TOTAL_BATTERY_CAPACITY
    percent_usage = next_demand / TOTAL_BATTERY_CAPACITY
    next_soc = s.battery_charge + percent_generation - percent_usage

    return State(next_soc, a, next_demand, next_gen, None, None)


def create_sim_state_list(data_dict):
    states = []
    powergen_df = data_dict['renewable']
    diesel_df = data_dict['diesel']
    demand_df = data_dict['demand']
    battery_df = data_dict['battery']

    for i in range(T):
        # TODO demand and gen prediction
        gen_on: bool = False
        if diesel_df['Power (W)'].iloc[i] > 0:
            gen_on = True
        s = State(battery_df['stateOfCharge (%)'].iloc[i],
                  gen_on,
                  demand_df['Power (W)'].iloc[i],
                  powergen_df['Power (W)'].iloc[i],
                  None,
                  None)
        states.append(s)
    return states


if __name__ == '__main__':
    main()

