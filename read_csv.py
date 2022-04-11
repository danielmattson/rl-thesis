import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

'''
read data from:
 https://sustainablesml.org/pages/systemList.php
 https://sustainablesml.org/pages/export.php

data:
 2 ECB power generation csv files (wind/solar) ->  combine to one total renewable power gen df 
 1 ECB battery soc file -> take SoC only
 1 Generator Room power demand csv file -> demand df
 1 diesel generator file -> take power output to later be mapped to indicator variable (on/off)
 
 how to create demand prediction?
 how to create power generation prediction?
'''

# TODO demand prediction?
# TODO power generation prediction?


def read_files():
    mypath = 'data/july2021/'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for file in onlyfiles:

    diesel_df = pd.read_csv(mypath + '2021-07-01_2021-07-31_diesel_generator_power.csv', comment='#').dropna()
    demand_df = pd.read_csv(mypath + '2021-07-01_2021-07-31_islandPowerUse.csv', comment='#').dropna()
    battery_df = pd.read_csv(mypath + '2021-07-01_2021-07-31_ecb_battery.csv', comment='#')[['Date', 'stateOfCharge (%)']].dropna()

    pvpower_df = pd.read_csv(mypath + '2021-07-01_2021-07-31_ecb_solarpowergen.csv', comment='#').dropna()
    windpower_df = pd.read_csv(mypath + '2021-07-01_2021-07-31_ecb_windpower.csv', comment='#').dropna()

    # combine wind + power generation to one table
    # only take rows on the 10 min intervals
    powergen_df = pvpower_df.loc[pvpower_df['Date'].isin(battery_df['Date'])].reset_index(drop=True)
    windpower_df = windpower_df.loc[windpower_df['Date'].isin(battery_df['Date'])].reset_index(drop=True)

    # rename to match so we can add later
    powergen_df.rename(columns={'pvHarvestPower (W)': 'Power (W)'}, inplace=True)
    windpower_df.rename(columns={'power (kW)': 'Power (W)'}, inplace=True)

    # convert to W
    windpower_df['Power (W)'] = (windpower_df['Power (W)'] * 1000)
    # drop unneeded columns
    windpower_df = windpower_df[['Date', 'Power (W)']]

    # add solar and wind power together
    powergen_df = pd.concat([powergen_df, windpower_df]).groupby('Date')['Power (W)'].sum().reset_index()

    # battery % is only measured every 10min, only consider rows that line up with battery readings
    demand_df = demand_df.loc[demand_df['Date'].isin(battery_df['Date'])].sort_values(by='Date').reset_index(drop=True)
    diesel_df = diesel_df.loc[diesel_df['Date'].isin(battery_df['Date'])].sort_values(by='Date').reset_index(drop=True)
    battery_df = battery_df.sort_values(by='Date').reset_index(drop=True)

    demand_df.rename(columns={'totalRealPower (W)': 'Power (W)'}, inplace=True)
    diesel_df.rename(columns={'totalRealPower (W)': 'Power (W)'}, inplace=True)

    # print_df(powergen_df, 'powergen')
    # print_df(demand_df, 'demand')
    # print_df(diesel_df, 'diesel')
    # print_df(battery_df, 'battery')

    # powervals = diesel_df[diesel_df['totalRealPower (W)'] != 0]['totalRealPower (W)']
    # avgdieseloutput = powervals.mean()
    # print(avgdieseloutput)

    # return {'renewable': powergen_df,
    #         'diesel': diesel_df,
    #         'demand': demand_df,
    #         'battery': battery_df}
    # 1008 rows is 7 days, 144 rows is 1 day
    endi = 1008
    return {'renewable': powergen_df.iloc[0:endi],
            'diesel': diesel_df.iloc[0:endi],
            'demand': demand_df.iloc[0:endi],
            'battery': battery_df.iloc[0:endi]}


def print_df(df, name='df'):
    print(f'{name}:\n {df}')
