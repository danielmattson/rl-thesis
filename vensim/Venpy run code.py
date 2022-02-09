# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:45:02 2022

@author: rg1070
"""

import venpy
import os
import timeit
import pandas as pd
import numpy as np

##Setup external file as dataframes
Data = pd.read_excel(
    r'vensim/Data.xlsx',
    sheet_name='Data')
Points_of_comparison = pd.read_excel(
    r'vensim/Data.xlsx',
    sheet_name='Points of comparison')
Power_Table = pd.read_excel(
    r'vensim/Data.xlsx',
    sheet_name='Power Table')
print(Data)
print(Points_of_comparison)
print(Power_Table)

##Seting up the home directory
# os.chdir("C:/Users/rg1070/Box/Shoals EWC microgrid project/Vensim models/SML Vensim Model/Model V.10_in PythonR")
print("Current Working Directory ", os.getcwd())

##Set up the model
# model = venpy.load('Shoal Island Microgrid System2.vpm')
model = venpy.load('Shoal Island Microgrid System.mdl')

# model.vtype
parameters = model.vtype
# print(parameters)
print("Module efficiency:", model['Module efficiency'])
print("DC to AC efficiency:", model['DC to AC efficiency'])
print("Number of Batteries:", model['Number of Batteries'])

model['DC to AC efficiency'] = 0.94

start = timeit.timeit()
model.run(runname='First Run')
end = timeit.timeit()
print(end - start)

model.result()

model.result(names=['Battery bank level percentage'])
model.result(names=['PV Generation'])
model.result(names=['Diesel electricity generation'])

# Model accuracy
# Battery state of charge
x_value1 = Points_of_comparison["State of charge% Real"]
y_value1 = model.result(['Battery bank level percentage'])
y_value1 = y_value1.squeeze()
correlation_matrix1 = np.corrcoef(x_value1, y_value1)
correlation_xy1 = correlation_matrix1[0, 1]
r_squared1 = correlation_xy1 ** 2
print(r_squared1)
MSE1 = np.square(np.subtract(x_value1, y_value1)).mean()
print(MSE1)

# PV generations
x_value2 = Points_of_comparison["Total Solar Generated Real (KW)"]
y_value2 = model.result(['PV Generation'])
y_value2 = y_value2.squeeze()
correlation_matrix2 = np.corrcoef(x_value2, y_value2)
correlation_xy2 = correlation_matrix2[0, 1]
r_squared2 = correlation_xy2 ** 2
print(r_squared2)
MSE2 = np.square(np.subtract(x_value2, y_value2)).mean()
print(MSE2)

# Diesel electricity generation
x_value3 = Points_of_comparison["Gen_totalRealPower (KW)"]
y_value3 = model.result(['Diesel electricity generation'])
y_value3 = y_value3.squeeze()
correlation_matrix3 = np.corrcoef(x_value3, y_value3)
correlation_xy3 = correlation_matrix3[0, 1]
r_squared3 = correlation_xy3 ** 2
print(r_squared3)
MSE3 = np.square(np.subtract(x_value3, y_value3)).mean()
print(MSE3)

###############################################################################
##DC2AC calibration 
DC2AC = pd.DataFrame(
    columns=['DC to AC efficiency', 'SOC MSE', 'SOC R2', 'PV Gen MSE', 'PV Gen R2', 'Gen Pow MSE', 'Gen Pow R2', ],
    index=pd.Series(range(1, 22)))

for x in range(0, 21):
    print((x / 100) + 0.8)
    model['DC to AC efficiency'] = ((x) / 100) + 0.8
    model.run(runname='First Run')

    # Model accuracy
    # Battery state of charge
    # Model accuracy
    # Battery state of charge
    x_value1 = Points_of_comparison["State of charge% Real"]
    y_value1 = model.result(['Battery bank level percentage'])
    y_value1 = y_value1.squeeze()
    correlation_matrix1 = np.corrcoef(x_value1, y_value1)
    correlation_xy1 = correlation_matrix1[0, 1]
    r_squared1 = correlation_xy1 ** 2
    print(r_squared1)
    MSE1 = np.square(np.subtract(x_value1, y_value1)).mean()
    print(MSE1)

    # PV generations
    x_value2 = Points_of_comparison["Total Solar Generated Real (KW)"]
    y_value2 = model.result(['PV Generation'])
    y_value2 = y_value2.squeeze()
    correlation_matrix2 = np.corrcoef(x_value2, y_value2)
    correlation_xy2 = correlation_matrix2[0, 1]
    r_squared2 = correlation_xy2 ** 2
    print(r_squared2)
    MSE2 = np.square(np.subtract(x_value2, y_value2)).mean()
    print(MSE2)

    # Diesel electricity generation
    x_value3 = Points_of_comparison["Gen_totalRealPower (KW)"]
    y_value3 = model.result(['Diesel electricity generation'])
    y_value3 = y_value3.squeeze()
    correlation_matrix3 = np.corrcoef(x_value3, y_value3)
    correlation_xy3 = correlation_matrix3[0, 1]
    r_squared3 = correlation_xy3 ** 2
    print(r_squared3)
    MSE3 = np.square(np.subtract(x_value3, y_value3)).mean()
    print(MSE3)

    DC2AC.iloc[x, 0] = (((x) / 100) + 0.8)
    DC2AC.iloc[x, 1] = MSE1
    DC2AC.iloc[x, 2] = r_squared1
    DC2AC.iloc[x, 3] = MSE2
    DC2AC.iloc[x, 4] = r_squared2
    DC2AC.iloc[x, 5] = MSE3
    DC2AC.iloc[x, 6] = r_squared3

print(DC2AC)

## Export results
DC2AC.to_csv(
    r'C:\Users\rg1070\Box\Shoals EWC microgrid project\Vensim models\SML Vensim Model\Model V.10_in PythonR\DC2AC.csv',
    index=True)

###############################################################################
##Module efficiency calibration
model['DC to AC efficiency'] = 0.94  ## defined based on previous part's results

ME = pd.DataFrame(
    columns=['DC to AC efficiency', 'SOC MSE', 'SOC R2', 'PV Gen MSE', 'PV Gen R2', 'Gen Pow MSE', 'Gen Pow R2', ],
    index=pd.Series(range(1, 9)))

for x in range(13, 21):
    print(x / 100)
    model['Module efficiency'] = x / 100
    model.run(runname='First Run')

    # Model accuracy
    # Battery state of charge
    # Model accuracy
    # Battery state of charge
    x_value1 = Points_of_comparison["State of charge% Real"]
    y_value1 = model.result(['Battery bank level percentage'])
    y_value1 = y_value1.squeeze()
    correlation_matrix1 = np.corrcoef(x_value1, y_value1)
    correlation_xy1 = correlation_matrix1[0, 1]
    r_squared1 = correlation_xy1 ** 2
    print(r_squared1)
    MSE1 = np.square(np.subtract(x_value1, y_value1)).mean()
    print(MSE1)

    # PV generations
    x_value2 = Points_of_comparison["Total Solar Generated Real (KW)"]
    y_value2 = model.result(['PV Generation'])
    y_value2 = y_value2.squeeze()
    correlation_matrix2 = np.corrcoef(x_value2, y_value2)
    correlation_xy2 = correlation_matrix2[0, 1]
    r_squared2 = correlation_xy2 ** 2
    print(r_squared2)
    MSE2 = np.square(np.subtract(x_value2, y_value2)).mean()
    print(MSE2)

    # Diesel electricity generation
    x_value3 = Points_of_comparison["Gen_totalRealPower (KW)"]
    y_value3 = model.result(['Diesel electricity generation'])
    y_value3 = y_value3.squeeze()
    correlation_matrix3 = np.corrcoef(x_value3, y_value3)
    correlation_xy3 = correlation_matrix3[0, 1]
    r_squared3 = correlation_xy3 ** 2
    print(r_squared3)
    MSE3 = np.square(np.subtract(x_value3, y_value3)).mean()
    print(MSE3)

    ME.iloc[x - 13, 0] = (x / 100)
    ME.iloc[x - 13, 1] = MSE1
    ME.iloc[x - 13, 2] = r_squared1
    ME.iloc[x - 13, 3] = MSE2
    ME.iloc[x - 13, 4] = r_squared2
    ME.iloc[x - 13, 5] = MSE3
    ME.iloc[x - 13, 6] = r_squared3

print(ME)
## Export results
ME.to_csv(
    r'C:\Users\rg1070\Box\Shoals EWC microgrid project\Vensim models\SML Vensim Model\Model V.10_in PythonR\ME.csv',
    index=True)

###############################################################################
##Battery size optimization

model['Module efficiency'] = 0.15  ## defined based on previous part's results

BTTOPT = pd.DataFrame(
    columns=['Numbr of batteries', 'Battery capacity', 'Fossil fuel energy consumption', 'System costs'],
    index=pd.Series(range(1, 116)))

for i in range(40, 151):
    print(i, 'units of battery, equal to:', i * 7.5, 'kWh of capacity')
    model['Number of Batteries'] = i
    model.run(runname='First Run')

    a = model.result(['"Cumulative diesel consumption from the generator (kWh)"'])
    b = model.result(['Cumulative system costs'])

    BTTOPT.iloc[i - 40, 0] = i
    BTTOPT.iloc[i - 40, 1] = i * 7.5
    BTTOPT.iloc[i - 40, 2] = a.iloc[103679, 0]
    BTTOPT.iloc[i - 40, 3] = b.iloc[103679, 0]

BTTOPT.to_csv(
    r'C:\Users\rg1070\Box\Shoals EWC microgrid project\Vensim models\SML Vensim Model\Model V.10_in PythonR\Battery optimization.csv',
    index=True)

###############################################################################
##Panel number optimization   

model['Number of Batteries'] = 300

PNLOPT = pd.DataFrame(columns=['Numbr of panels', 'Fossil fuel energy consumption', 'System costs'],
                      index=pd.Series(range(1, 202)))

for i in range(0, 201):
    print(i, 'Number op panels equal to', i + 233)
    model['Number of solar panel installed'] = i + 233
    model.run(runname='First Run')

    a = model.result(['"Cumulative diesel consumption from the generator (kWh)"'])
    b = model.result(['Cumulative system costs'])

    PNLOPT.iloc[i, 0] = i + 233
    PNLOPT.iloc[i, 1] = a.iloc[103679, 0]
    PNLOPT.iloc[i, 2] = b.iloc[103679, 0]

PNLOPT.to_csv(
    r'C:\Users\rg1070\Box\Shoals EWC microgrid project\Vensim models\SML Vensim Model\Model V.10_in PythonR\Panel number optimization.csv',
    index=True)

###############################################################################
##Panel number and battery size optimization

model['DC to AC efficiency'] = 0.94  ## defined based on previous part's results
model['Module efficiency'] = 0.15  ## defined based on previous part's results
model['Number of Batteries'] = 300
model['Number of solar panel installed'] = 233

TWO_WAY_OPT = pd.DataFrame(columns=['Numbr of panels', 'Numbr of batteries',
                                    'Battery capacity', 'Fossil fuel energy consumption', 'System costs'],
                           index=pd.Series(range(1, 11212)))

for i in range(0, 101):
    model['Number of solar panel installed'] = i + 233

    for j in range(40, 151):
        print('Number of panels equal to', i + 233)
        print(j, 'units of battery, equal to:', j * 7.5, 'kWh of capacity')
        model['Number of Batteries'] = j

        model.run(runname='First Run')

        a = model.result(['"Cumulative diesel consumption from the generator (kWh)"'])
        b = model.result(['Cumulative system costs'])

        TWO_WAY_OPT.iloc[(i * 111) + j - 40, 0] = j
        TWO_WAY_OPT.iloc[(i * 111) + j - 40, 1] = i
        TWO_WAY_OPT.iloc[(i * 111) + j - 40, 2] = i * 7.5
        TWO_WAY_OPT.iloc[(i * 111) + j - 40, 3] = a.iloc[103679, 0]
        TWO_WAY_OPT.iloc[(i * 111) + j - 40, 4] = b.iloc[103679, 0]

TWO_WAY_OPT.to_csv(r'C:\Users\rg1070\Box\Shoals EWC microgrid project\Vensim models\SML Vensim Model\Model V.10_in '
                   r'PythonR\Panel number and Battery capacity optimization.csv', index=True)

### test

TWO_WAY_OPT = pd.DataFrame(columns=['value'],
                           index=pd.Series(range(1, 11212)))
for i in range(0, 101):

    for j in range(40, 151):
        print((i * 111) + j - 40)
        TWO_WAY_OPT.iloc[(i * 111) + j - 40, 0] = (i * 111) + j - 40

'''
#example
model = venpy.load('simple.vpm')
model.vtype
model['Characteristic Time']

print("Characteristic Time is initially:" , model['Characteristic Time'])
model['Characteristic Time'] = 5
print("Characteristic Time is initially:" , model['Characteristic Time'])

model.run()
model.result()
model.result(names=['Stock'])

print(model['Stock'])
'''
