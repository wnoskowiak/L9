import openpyxl   
import pandas as pd
from pandas import read_excel
import matplotlib.pyplot as plt

raw_data=pd.read_excel("nitrogen.xlsx", engine = 'openpyxl') 
data = pd.DataFrame(raw_data)
print(data)
plt.plot(data['time'].to_numpy()[2:],data['inverted'].to_numpy()[2:])
plt.show()


print(data.to_numpy())
