import openpyxl   
import pandas as pd
import numpy as np
from pandas import read_excel
import matplotlib.pyplot as plt
import scipy.optimize as scp

def f(x,a,b):
    return a*np.exp(-x/b)
    

raw_data=pd.read_excel("air2.xlsx", engine = 'openpyxl') 
data = pd.DataFrame(raw_data)
data = data.drop([0,1]).reset_index(drop=True)

time_column = data.columns[2]
ch1_column = data.columns[3]
data_column = data.columns[-1]

plt.plot(data[time_column],data[ch1_column])
plt.plot(data[time_column],data[data_column], color='r')
plt.show()

tester = []
for i in range(len(data[ch1_column])):
    tester.append(data[ch1_column].to_numpy()[i]-data[ch1_column].to_numpy()[i-1])

thershold =  (np.max(tester)-np.average(tester))/3.5
print(thershold)

plt.plot(data[time_column], tester)
plt.axhline(thershold, color='r')
plt.show()

points_candidates = np.argwhere(tester>thershold)
points_candidates = np.append(points_candidates,[[(len(data[ch1_column])-1)]], axis=0)
#points_candidates.append()

i = 0
points = []
while i<(len(points_candidates)-1):
    distance = points_candidates[i+1][0]-points_candidates[i][0]
    if distance < 200:
        if data[ch1_column][points_candidates[i][0]]>data[ch1_column][points_candidates[i+1][0]]:
            points.append(points_candidates[i][0])
        else:
            points.append(points_candidates[i+1][0])
        i = i + 1

    else:
        points.append(points_candidates[i][0])

    i = i + 1

points.append(len(data[ch1_column])-1)
#print(points)
#print(points_candidates)



plt.plot(data[time_column],data[data_column])


for i in range(len(points)):
    plt.axvline(x= data[time_column][points[i]], color = 'r')

"""

for i in range(len(points_candidates)):
    plt.axvline(x= data[time_column][points_candidates[i][0]], color = 'g')

"""

plt.show()

thaus = []

#for i in [10]:

for i in range(len(points)-1):

    indexes = np.arange(points[i],points[i+1])[:-20]
    val_between_imp = data[data_column].to_numpy()[indexes]
    peak = np.argmax(val_between_imp)

    times_fit = np.array(data[time_column ].to_numpy()[indexes][peak:] - data[time_column ].to_numpy()[indexes][peak:][0],dtype=np.float32)
    val_fit = np.array((data[data_column].to_numpy()[indexes][peak:]-np.average(data[data_column].to_numpy()[indexes][peak:][-20:])),dtype=np.float32)
    a_init = val_fit[0]
    b_init = 0.00000014
    #b_init = -times_fit[80]/np.log(val_fit[80]/a_init)

    params, pcov = scp.curve_fit(f, times_fit, val_fit, [a_init,b_init])
    err = np.sqrt(np.diag(pcov))

    thaus.append(params[1])
    #print(params[1])
    
    #"""
    """
    %matplotlib widget
    plt.plot(times_fit,val_fit)
    plt.plot(times_fit, f(times_fit, params[0], params[1]), color = 'r')
    """
    #"""
    
print(thaus)
print(np.average(thaus),np.std(thaus))

