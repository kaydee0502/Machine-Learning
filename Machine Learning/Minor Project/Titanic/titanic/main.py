#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:26:33 2020

@author: kaydee
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

data = pd.read_csv("train.csv")

def o(x):
    return x==1



em = data.groupby("Survived")["Pclass"]
sur = em.get_group(1)
dsur = em.get_group(0)

fc = [sur[sur == 1].count(),dsur[dsur == 1].count()]
sc = [sur[sur == 2].count(),dsur[dsur == 2].count()]
tc = [sur[sur == 3].count(),dsur[dsur == 3].count()]
fix,ax= plt.subplots()
ax.bar(["sur","dsur"],fc,color="r",label="1st Class")
ax.bar(["sur","dsur"],sc,color="g",bottom=fc,label="2nd Class")
ax.bar(["sur","dsur"],tc,color="b",bottom=np.array(fc)+np.array(sc),label="3rd Class")
ax.legend()
ticks_loc = ax.get_xticks()
ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ax.set_xticklabels(["survived","did not survived"])

plt.show()