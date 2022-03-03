import random
from enum import Enum, unique
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
from typing import Dict
import tensorflow
from keras import Model
from keras import Sequential

import read_csv

T = 0

GENERATOR_REWARD = 10
BAD_SOC_COST = -1000

AVG_GENERATOR_OUTPUT = 13000
ACTIONS = {0, 1}

# using data from 2017 SEI report, page 71. ECB battery max capacity is 700 Ah
# in more recent CSVs(july 2021), voltage is ~50 V
# Wh = V * Ah = 50 * 700 = 35000 Wh = 35 kWh
TOTAL_BATTERY_CAPACITY = 35000


V_TILDE = 0


@unique
class Policy(Enum):
    RANDOM = 0
    GEN_WHEN_UNDER_70 = 1
    GEN_WHEN_UNDER_50 = 2
    GEN_WHEN_UNDER_25 = 3
    GEN_WHEN_UNDER_70_AND_HIGH_DEMAND = 4
    PVI = 5


# TODO optimize by replacing class with dictionary of numpy arrays, one for each variable
#  or big array w one dimension to choose a variable?
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


def policy(s, t, policy):
    if policy == Policy.RANDOM:
        return random.choice([True, False])
    elif policy == Policy.GEN_WHEN_UNDER_70:
        return s.battery_charge < 70
    elif policy == Policy.GEN_WHEN_UNDER_50:
        return s.battery_charge < 50
    elif policy == Policy.GEN_WHEN_UNDER_25:
        return s.battery_charge < 25
    elif policy == Policy.GEN_WHEN_UNDER_70_AND_HIGH_DEMAND:
        return s.battery_charge < 70 and s.demand > s.generation
    elif policy == Policy.PVI:
        return fvi_policy(s, t, V_TILDE)
    else:
        return None


# TODO should data_states be referred to as a trajectory?
def pvi_v_tilde(data_states):
    # dimension of feature function
    K = 2

    # amount of states to sample from data
    S_hat_size = 1000

    # linear feature function
    # TODO how to handle multiple variables from each state? need more dimensions in phi?
    # TODO what is a good feature function?
    phi = lambda s: np.array([1, s.battery_charge])

    # weights for fitting model
    w = np.zeros([T + 1, K])

    # samples of "true" value function
    v_hat = np.zeros([T + 1, S_hat_size])

    # TODO make sample evenly spaced to get data from all times of month, this takes random values
    S_hat = random.sample(data_states, S_hat_size)

    # for fitting model
    A = np.array([phi(s) for s in S_hat])

    # TODO make work
    # calculate w and v_hat values for each state
    for t in reversed(range(T - 1)):
        for state_index in range(S_hat_size):
            state = S_hat[state_index]
            rewards_by_action = np.zeros([len(ACTIONS)])
            for action in ACTIONS:
                # compare possible actions and the reward from v_tilde(phi*w) for next state
                rewards_by_action[action] = r(state, action) + phi(f(state, action, t)).T @ w[t + 1]

                # avg across disturbances
                # rewards_by_action[action] = np.average(rewards_by_action[action])
            # store max value of each state in v_hat
            v_hat[t][state_index] = np.max(rewards_by_action)

        # calculate w by minimizing error between phi.T @ w and v_hat
        w[t] = np.linalg.lstsq(A, v_hat[t], rcond=0)[0]
        # print(f'ws {w[t]} at time {t}')

    # use final weights
    w = w[0]
    print(f'final w {w}')

    # approximated value function
    v_tilde = lambda s: phi(s).T @ w
    # v_tilde_v = np.vectorize(v_tilde)
    return v_tilde


# once we have w, use this to choose actions given a state
# given state, choose action according to FVI greedy policy
# TODO make work
def fvi_policy(state, t, approx_value_function):
    reward_by_action = np.zeros([2])

    # compute reward for each action
    for action in ACTIONS:
        next_state_value = approx_value_function(f(state, action, t))
        print(f'next state approx value: {next_state_value}')
        reward_by_action[action] = r(state, action) + next_state_value
    # return action that gives max reward
    return np.argmax(reward_by_action)


def init_policy(policy_choice, data):
    if policy_choice == Policy.PVI:
        # approximate value function with PVI method
        global V_TILDE
        V_TILDE = pvi_v_tilde(data)
    else:
        # other policies will be added later?
        pass


def sim_rl():
    data_dict: Dict[str, pd.DataFrame] = read_csv.read_files()
    global T
    T = len(data_dict['demand']) - 1
    # len() - 1 because f() depends on next state from data


    # TODO should data_states be referred to as a trajectory?
    global data_states
    data_states = create_data_state_list(data_dict)

    ###### choose policy here
    policy_choice = Policy.PVI
    init_policy(policy_choice, data_states)
    #########################

    M = 1
    for m in range(M):
        cumulative_reward = np.zeros([T])
        total_reward = 0
        states = [State(100, False, 0, 0, None, None)]
        for t in range(T - 1):
            s = states[t]

            action = policy(s, t, policy_choice)
            print(f'action {action} at time {t}')

            states.append(f(s, action, t))
            total_reward += r(s, action)
            cumulative_reward[t] = total_reward
        final_state = states[-1]
        total_reward += r(final_state, None)
        cumulative_reward[-1] = total_reward
        print(f'reward from policy: {total_reward}')

        END_X = T
        x = range(END_X)
        pp.figure(1)
        pp.plot(x, [states[i].battery_charge for i in range(END_X)])
        # pp.title('Battery SoC over Time - Lower bound 70%')
        # pp.title('Battery SoC over Time - Lower bound 50%')
        pp.title('Battery SoC over Time - Lower bound 25%')
        # pp.title('Battery SoC over Time - Lower bound 70% and high demand')
        pp.xlabel('Time (10 min intervals)')
        pp.ylabel('Battery State of Charge (%)')

        pp.figure(2)
        pp.plot(x, cumulative_reward[0:END_X])
        # pp.title('Cumulative Reward over Time - Lower bound 70%')
        # pp.title('Cumulative Reward over Time - Lower bound 50%')
        pp.title('Cumulative Reward over Time - Lower bound 25%')
        # pp.title('Cumulative Reward over Time - Lower bound 70% and high demand')
        pp.xlabel('Time (10 min intervals)')
        pp.ylabel('Cumulative Reward')

        # pp.figure(3)
        # pp.plot(x, [states[i].generation for i in range(end_x)])
        # pp.title('Renewable Power Generation over Time')
        # pp.xlabel('Time (10 min intervals)')
        # pp.ylabel('Power Generation (W)')

        pp.show()


def r(s: State, a):
    reward = GENERATOR_REWARD if s.is_generator_on else 0
    return reward + BAD_SOC_COST if s.battery_charge < 70 else reward


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

