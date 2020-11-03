# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:36:57 2020

@author: KayDee
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor 




data = pd.read_csv("train.csv")
desc = data.describe()
head = data.head(10)

model = DecisionTreeRegressor(random_state=1)
X = head[['MSSubClass','LotFrontage','LotArea','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','MasVnrArea','Fireplaces','GarageArea']]

bol = X['LotFrontage'].isnull()
print(X[bol])
X['LotFrontage'].fillna(X['LotFrontage'].mean(),inplace=True)

y = head['SalePrice']
trainer = model.fit(X,y)
pred = model.predict(X)
print(data[["SaleType","MSZoning"]])
print(pred)