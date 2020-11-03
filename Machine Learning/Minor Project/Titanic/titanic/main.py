#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:26:33 2020

@author: kaydee
"""
import pandas as pd

data = pd.read_csv("train.csv")

def o(x):
    return x==1

em = data.groupby("Embarked")["Survived"]
print(em.get_group("S"))
print(data.head())