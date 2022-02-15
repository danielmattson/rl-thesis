import random
from enum import Enum, unique
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
from typing import Dict

import read_csv

T = 0
GENERATOR_COST = -10
AVG_GENERATOR_OUTPUT = 13000


@unique
class Policy(Enum):
    RANDOM = 0
    GEN_WHEN_UNDER_70 = 1
    GEN_WHEN_UNDER_50 = 2
    GEN_WHEN_UNDER_25 = 3


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
    # init_vensim()
    sim_rl()


def init_vensim():
    pass


def policy(s, policy):
    if policy == Policy.RANDOM:
        return random.choice([True, False])
    elif policy == Policy.GEN_WHEN_UNDER_70:
        return s.battery_charge < 70
    elif policy == Policy.GEN_WHEN_UNDER_50:
        return s.battery_charge < 50
    elif policy == Policy.GEN_WHEN_UNDER_25:
        return s.battery_charge < 25
    else:
        return None


def sim_rl():
    data_dict: Dict[str, pd.DataFrame] = read_csv.read_files()
    global T
    T = len(data_dict['demand'])
    global data_states
    data_states = create_data_state_list(data_dict)
    M = 1

    for m in range(M):
        cumulative_reward = np.zeros([T])
        total_reward = 0
        states = [State(100, False, 0, 0, None, None)]
        for t in range(T - 1):
            s = states[t]
            ###### choose policy here
            action = policy(states[t], Policy.GEN_WHEN_UNDER_25)
            #########################

            states.append(f(s, action, t))
            total_reward += r(s, action)
            cumulative_reward[t] = total_reward
        final_state = states[-1]
        total_reward += r(final_state, None)
        cumulative_reward[-1] = total_reward
        print(f'reward from random policy: {total_reward}')

        END_X = 1000
        x = range(END_X)
        pp.figure(1)
        pp.plot(x, [states[i].battery_charge for i in range(END_X)])
        # pp.title('Battery SoC over Time - Lower bound 70%')
        # pp.title('Battery SoC over Time - Lower bound 50%')
        pp.title('Battery SoC over Time - Lower bound 25%')
        pp.xlabel('Time (10 min intervals)')
        pp.ylabel('Battery State of Charge (%)')

        pp.figure(2)
        pp.plot(x, cumulative_reward[0:END_X])
        # pp.title('Cumulative Reward over Time - Lower bound 70%')
        # pp.title('Cumulative Reward over Time - Lower bound 50%')
        pp.title('Cumulative Reward over Time - Lower bound 25%')
        pp.xlabel('Time (10 min intervals)')
        pp.ylabel('Cumulative Reward')

        # pp.figure(3)
        # pp.plot(x, [states[i].generation for i in range(end_x)])
        # pp.title('Renewable Power Generation over Time')
        # pp.xlabel('Time (10 min intervals)')
        # pp.ylabel('Power Generation (W)')

        pp.show()


def r(s: State, a):
    # if generator is left on
    if s.is_generator_on:
        return GENERATOR_COST
    else:
        # no cost to rely on renewable energy
        return 0


def f(s: State, a, t) -> State:
    next_data_state = data_states[t + 1]
    next_demand = next_data_state.demand
    next_gen = next_data_state.generation

    percent_generation = (next_gen + (AVG_GENERATOR_OUTPUT if s.is_generator_on else 0)) / TOTAL_BATTERY_CAPACITY
    percent_usage = next_demand / TOTAL_BATTERY_CAPACITY
    next_soc = s.battery_charge + percent_generation - percent_usage

    return State(next_soc, a, next_demand, next_gen, None, None)


def create_data_state_list(data_dict):
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

