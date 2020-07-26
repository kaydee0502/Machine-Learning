import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')

x = data.iloc[:,3:11].values
y = data.iloc[:,15].values

plt.scatter(y,x[:,7],color='red')
#plt.plot(x_train,rgrsr.predict(x_train),color='green')
plt.title("Cereals")
plt.xlabel("x")
plt.ylabel("y")
plt.show