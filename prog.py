import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp

def f(x,a,b,c):
    return ((a*np.exp(-x/b)) -c)

number = 7
test_samples = ['nitrogen','air1','air2','crone1','crone2','crone3','spark1','spark2']
names = ['odparowany azot', 'powietrze przed wyładowaniami', 'powietrze po wyładowaniach', 'słabe wyładowanie koronowe', 'średnie wyładowanie koronowe', 'silne wyłaodowanie koronowe', 'słabe wyładowanie iskrowe', 'mocne wyładowanie iskrowe']
raw_data=pd.read_excel(str(test_samples[number]+".xlsx"), engine = 'openpyxl') 
data = pd.DataFrame(raw_data)
data = data.drop([0,1]).reset_index(drop=True)


time_column = data.columns[2]
ch1_column = data.columns[3]
data_column = data.columns[-1]
#print(time_column,ch1_column,data_column)

tester = []
for i in range(len(data[ch1_column])):
    tester.append(data[ch1_column].to_numpy()[i]-data[ch1_column].to_numpy()[i-1])

thershold =  (np.max(tester)-np.average(tester))/3.5
#print(thershold)

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

int_indexes = np.arange(points[0],points[1])
temp_data = data[data_column].to_numpy()
average = temp_data[int_indexes]
res = []

for i in range(len(int_indexes)):
    temp = []
    for k in range(len(points)):
        try:
            temp.append(temp_data[points[k]+i])
        except:
            pass
    res.append(np.average(temp))

peak = np.argmax(res)

times_all = data[time_column ].to_numpy()[int_indexes]
cutoff = int(len(times_all))

t_0 = times_all[peak:][0]
times_fit = np.array(times_all[peak:] - t_0, dtype=np.float32)[:cutoff]
#times_fit = data[time_column ].to_numpy()[int_indexes][peak:]
val_fit = res[peak:][:cutoff]

a_init = val_fit[0]
b_init =  4.61052126e-07
c_init = -2.65097587e-04

params, pcov = scp.curve_fit(f, times_fit, val_fit, [a_init,b_init, c_init], method = 'lm' )
err = np.sqrt(np.diag(pcov))
print(params, err)

plt.figure(figsize=(8.5,5))
plt.plot(times_all[:-40], res[:-40])
plt.plot((times_fit+t_0)[:-40], f(times_fit, *params)[:-40], color = 'r')
plt.xlabel("czas [s]")
plt.ylabel("napięcie [V]")
plt.title(names[number])
#plt.rcParams["figure.figsize"] = (5,10)
plt.show()
