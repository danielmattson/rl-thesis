import random
from enum import Enum, unique

import keras.models
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
from typing import Dict
import tensorflow
from keras import Model
from keras import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from numpy import exp

import read_csv

T = 0
data_dict = dict()

GENERATOR_REWARD = -10
BAD_SOC_COST = -1000
CRITICAL_SOC_COST = -1000000

AVG_GENERATOR_OUTPUT = 13000
ACTIONS = {0, 1}

# using data from 2017 SEI report, page 71. ECB battery max capacity is 700 Ah
# in more recent CSVs(july 2021), voltage is ~50 V
# Wh = V * Ah = 50 * 700 = 35000 Wh = 35 kWh
TOTAL_BATTERY_CAPACITY = 35000

V_TILDE = 0

feature_function = {
    "linear": lambda s: np.array(
        [1, s.is_generator_on, s.battery_charge * s.is_generator_on, s.battery_charge * (1 - s.is_generator_on)]),
    "cubic_spline": lambda s: np.array(
        [1, s.is_generator_on,
         s.battery_charge * s.is_generator_on,
         s.battery_charge * (1 - s.is_generator_on),
         s.battery_charge ** 2,
         s.battery_charge ** 3,
         # max(0, s.battery_charge - 20) ** 3,
         # max(0, s.battery_charge - 30) ** 3,
         # max(0, s.battery_charge - 40) ** 3,
         max(0, s.battery_charge - 50) ** 3,
         max(0, s.battery_charge - 60) ** 3,
         max(0, s.battery_charge - 70) ** 3,
         max(0, s.battery_charge - 80) ** 3,
         max(0, s.battery_charge - 90) ** 3]),
    "linear_spline": lambda s: np.array(
        [1, s.is_generator_on,
         s.battery_charge * s.is_generator_on,
         s.battery_charge * (1 - s.is_generator_on),
         max(0, s.battery_charge - 20),
         max(0, s.battery_charge - 30),
         max(0, s.battery_charge - 40),
         max(0, s.battery_charge - 50),
         max(0, s.battery_charge - 60),
         max(0, s.battery_charge - 70),
         max(0, s.battery_charge - 80),
         max(0, s.battery_charge - 90)]),
    "rbf": lambda s: np.array(
        [s.is_generator_on,
         s.battery_charge * s.is_generator_on,
         s.battery_charge * (1 - s.is_generator_on),
         exp(-(s.battery_charge - 20) ** 2 / 50),
         exp(-(s.battery_charge - 40) ** 2 / 50),
         exp(-(s.battery_charge - 60) ** 2 / 50),
         exp(-(s.battery_charge - 75) ** 2 / 50),
         exp(-(s.battery_charge - 90) ** 2 / 50)])
}


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
    def __init__(self, SoC=0, is_gen_on=0, dem=0, gen=0, dem_pred=None, gen_pred=None):
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
    ret = None
    if policy == Policy.RANDOM:
        ret = random.choice([True, False])
    elif policy == Policy.GEN_WHEN_UNDER_70:
        ret = s.battery_charge < 70
    elif policy == Policy.GEN_WHEN_UNDER_50:
        ret = s.battery_charge < 50
    elif policy == Policy.GEN_WHEN_UNDER_25:
        ret = s.battery_charge < 25
    elif policy == Policy.GEN_WHEN_UNDER_70_AND_HIGH_DEMAND:
        ret = s.battery_charge < 70 and s.demand > s.generation
    elif policy == Policy.PVI:
        ret = pvi_policy(s, t, V_TILDE)
    else:
        ret = None
    return int(ret)


def init_nn():
    model = Sequential()
    model.add(Dense(2, activation='relu', input_dim=2))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    # model.summary()

    pd.set_option('display.max_columns', None)

    # create x, y for training
    battery_df = data_dict['battery']
    powergen_df = data_dict['renewable']
    demand_df = data_dict['demand']
    diesel_df = data_dict['diesel']

    powergen_df2 = powergen_df.rename(columns={'Power (W)': 'gen Power (W)'})
    demand_df2 = demand_df.rename(columns={'Power (W)': 'demand Power (W)'})
    diesel_df2 = diesel_df.rename(columns={'Power (W)': 'diesel Power (W)'})

    # map power > 0 to be 1 to represent on/off categorical
    # using one hot encoding because it might be faster to train model?
    onoff_df = pd.cut(diesel_df2['diesel Power (W)'], bins=[-1, 1, 999999], labels=[0, 1])
    # print(f'onoff:\n{onoff_df}')
    # encoded = to_categorical(onoff_df)
    # print(f'encoded:\n{encoded}')
    # diesel_df2.insert(2, 'generator on', encoded)
    diesel_df2.insert(2, 'generator on', onoff_df)

    # merge dfs to create one that is: soc | generation | demand | gen_on
    tmp = pd.merge(battery_df, powergen_df2, on='Date', how='left')
    tmp = pd.merge(tmp, demand_df2, on='Date', how='left')
    x = pd.merge(tmp, diesel_df2, on='Date', how='left')
    x.drop('Date', axis=1, inplace=True)
    x.drop('gen Power (W)', axis=1, inplace=True)
    x.drop('demand Power (W)', axis=1, inplace=True)
    x.drop('diesel Power (W)', axis=1, inplace=True)
    # x.rename(columns={'diesel Power (W)': 'generator on'}, inplace=True)
    # print(f'x:\n{x.to_string()}')

    y = pd.DataFrame(0.0, index=np.arange(x.shape[0]), columns=['value'])
    # print(f'y:\n{y}')

    model.fit(x, y, verbose=0, epochs=1, steps_per_epoch=1)
    return model, x, y


def train_nn(nn, x, y, S_hat):
    for t in reversed(range(T - 1)):
        percent_complete = ((T-1-t) / (T-1)*100)
        if int(percent_complete*10) % 5 == 0:
            print(f'starting iteration {T-1-t} out of {T - 1}: {percent_complete:.2f}% complete')
        for state_index in range(len(S_hat)):
            state: State = S_hat[state_index]
            rewards_by_action = np.zeros([len(ACTIONS)])
            for action in ACTIONS:
                # compare possible actions and the reward from v_tilde (nn prediction) for next state
                next_state = f(state, action, t)
                next_state_df = pd.DataFrame([[next_state.battery_charge, next_state.is_generator_on]],
                                        columns=['stateOfCharge (%)', 'generator on'])
                v_approx = nn(next_state_df.values)[0][0]

                # print(f'v_approx: {v_approx}')
                rewards_by_action[action] = r(state, action) + v_approx

            # update optimal value of each state in y df
            y.at[state_index, 'value'] = np.max(rewards_by_action)

        # refit keras to predict value functions of next state
        nn.fit(x, y, steps_per_epoch=1, epochs=1, verbose=0)

    return nn


def pvi_v_tilde(data_states):
    # amount of states to sample from data
    S_hat_size = 50
    # S_hat_size = int(len(data_states) / 10)

    # nn, x, y = init_nn()

    phi = feature_function['linear_spline']
    # dimension of feature function
    K = phi(State()).shape[0]

    # weights for fitting model
    w = np.zeros([T + 1, K])

    # samples of "true" value function
    v_hat = np.zeros([T + 1, S_hat_size])

    # TODO make sample evenly spaced to get data from all times of month? this takes random values
    S_hat = random.sample(data_states, S_hat_size)

    # for minimizing error by least squares
    A = np.array([phi(s) for s in S_hat])

    # nn = train_nn(nn, x, y, S_hat)
    # nn = keras.models.load_model('trained_nn')
    # calculate w and v_hat values for each state
    for t in reversed(range(T - 1)):
        # print(f'starting iteration {T-1-t} out of {T - 1}: {((T-1-t) / (T-1)*100):.2f}% complete')
        for state_index in range(S_hat_size):
            state: State = S_hat[state_index]
            rewards_by_action = np.zeros([len(ACTIONS)])
            for action in ACTIONS:
                # compare possible actions and the reward from v_tilde  (phi@w) for next state
                rewards_by_action[action] = r(state, action) + phi(f(state, action, t)).T @ w[t + 1]

            # store max value of each state in v_hat
            v_hat[t][state_index] = np.max(rewards_by_action)

        # calculate w by minimizing error between phi.T @ w and v_hat
        w[t] = np.linalg.lstsq(A, v_hat[t], rcond=0)[0]
        # print(f'ws {w[t]} at time {t}')

    # use final weights
    w = w[0]
    # print(f'\n\nfinal w: {w}\n\n')

    # approximated value function
    v_tilde = lambda s: phi(s).T @ w
    # nn.save('trained_nn_epochtest')
    # v_tilde = lambda s: \
    #     nn(pd.DataFrame([[s.battery_charge, float(s.is_generator_on)]], columns=['stateOfCharge (%)', 'generator on']).values)[0][0]
    return v_tilde


# once we have v_tilde:
# given state, choose action according to PVI greedy policy
def pvi_policy(state, t, approx_value_function):
    reward_by_action = np.zeros([2])

    # compute reward for each action
    for action in ACTIONS:
        next_state_value = approx_value_function(f(state, action, t))
        # print(f'next state approx value: {next_state_value}')
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


# def avg_policy(battery_charge_col, policy_choice):
#     # using given policy, return a list of the most likely action across all time steps
#     SOC_STEPS = len(battery_charge_col)
#     T_STEPS = 50
#     policy_by_t = np.zeros([T_STEPS, SOC_STEPS])
#     # for t in np.arange(0, T, T/T_STEPS):
#     for t in np.trunc(np.linspace(0, T, T_STEPS)):
#         for index, soc in enumerate(battery_charge_col):
#             policy_by_t[t][index] = policy(State(SoC=soc), 0, policy_choice)
#
#     # average all 0/1 values across each time step, then round to map to 0 or 1
#     avg_action = np.round(np.average(policy_by_t, axis=1))
#     return avg_action


def sim_rl():
    global data_dict
    data_dict = read_csv.read_files()
    # data_dict: Dict[str, pd.DataFrame] = read_csv.read_files()
    global T
    T = len(data_dict['demand']) - 1
    # len() - 1 because f() depends on next state from data

    global data_states
    data_states = create_data_state_list(data_dict)

    ###### choose policy here
    policy_choice = Policy.PVI
    init_policy(policy_choice, data_states)
    #########################

    M = 1
    for m in range(M):
        cumulative_cost = np.zeros([T])
        total_reward = 0
        gen_was_run = False
        states = [State(100, False, 0, 0, None, None)]
        for t in range(T - 1):
            s = states[t]

            action = policy(s, t, policy_choice)
            if action == 1:
                # print(f'generator run at time {t}')
                gen_was_run = True
            # else:
            #     print(f'.')

            states.append(f(s, action, t))
            total_reward += r(s, action)
            cumulative_cost[t] = -total_reward
        final_state = states[-1]
        total_reward += r(final_state, None)
        cumulative_cost[-1] = -total_reward
        print(f'total reward from policy: {total_reward}')
        if not gen_was_run:
            print(f'generator was never run')

        END_X = T
        x = range(END_X)
        pp.figure(1)
        pp.plot(x, [states[i].battery_charge for i in range(END_X)])
        pp.title('Battery SoC over Time')
        pp.xlabel('Time (10 min intervals)')
        pp.ylabel('Battery State of Charge (%)')

        pp.figure(2)
        pp.plot(x, cumulative_cost[0:END_X])
        pp.title('Cumulative Cost over Time')
        pp.xlabel('Time (10 min intervals)')
        pp.ylabel('Cumulative Cost')

        if policy_choice == Policy.PVI:
            pp.figure(3)
            pp.plot(x, [V_TILDE(states[i]) for i in range(END_X)])
            pp.title('v(s) over Time')
            pp.xlabel('Time (10 min intervals)')
            pp.ylabel('v(s)')

            # value function with generator on/off, x axis is battery SoC
            pp.figure(4)
            states_gen_on, = pp.plot(range(101), [V_TILDE(State(SoC=i, is_gen_on=1)) for i in range(101)], c='r')
            states_gen_off, = pp.plot(range(101), [V_TILDE(State(SoC=i, is_gen_on=0)) for i in range(101)], c='g')
            pp.legend([states_gen_on, states_gen_off], ['Generator on', 'Generator off'])
            pp.title('Value Function over SoC')
            pp.xlabel('SoC (%)')
            pp.ylabel('v(s)')

        # visualize policy
        # battery_charge_col = np.arange(0, 105, 5)
        # onoff_col = avg_policy(battery_charge_col, policy_choice)
        # policy_df = pd.DataFrame(zip(battery_charge_col, onoff_col), columns=['battery charge (%)', 'on/off'])
        # print(f'policy: \n\n {policy_df}')

        # pp.figure(3)
        # pp.plot(x, [states[i].generation for i in range(end_x)])
        # pp.title('Renewable Power Generation over Time')
        # pp.xlabel('Time (10 min intervals)')
        # pp.ylabel('Power Generation (W)')

        pp.show()


def r(s: State, a):
    reward = GENERATOR_REWARD if s.is_generator_on else 0
    reward = reward + BAD_SOC_COST if s.battery_charge < 70 else reward
    reward = reward + CRITICAL_SOC_COST if s.battery_charge < 20 else reward
    return reward


def f(s: State, a, t) -> State:
    next_data_state = data_states[t + 1]
    next_demand = next_data_state.demand
    next_gen = next_data_state.generation

    percent_generation = (next_gen + (AVG_GENERATOR_OUTPUT if s.is_generator_on else 0)) / TOTAL_BATTERY_CAPACITY
    percent_usage = next_demand / TOTAL_BATTERY_CAPACITY

    next_soc = min(100, s.battery_charge + percent_generation - percent_usage)
    if next_soc < 0:
        next_soc = 0

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
