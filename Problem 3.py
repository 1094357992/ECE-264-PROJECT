import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

filename = "NYC_Bicycle_Counts_2016_Corrected.csv"
dataset = pd.read_csv(filename)
row = dataset.shape[0]

for i in range(5, 10):
    for j in range(row):
        comma = dataset.iat[j, i].split(",")
        toInt = ""
        for x in comma:
            toInt += x
        dataset.iat[j, i] = int(toInt)

for i in range(2, 5):
    for j in range(row):
        dataset.iat[j, i] = float(dataset.iat[j, i])
        
# Find average number of bicyclists on each bridges per day in the week
def avem(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    m = 0
    m1 = 0
    for i in range(len(day)):
        if day[i] == "Monday":
            m += 1
            m1 += total[i]
    avem = m1 / m
    return avem
def avet(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    t = 0
    t1 = 0
    for i in range(len(day)):
        if day[i] == 'Tuesday':
            t += 1
            t1 += total[i]
    avet = t1 / t
    return avet
def avew(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    t = 0
    t1 = 0
    for i in range(len(day)):
        if day[i] == 'Wednesday':
            t += 1
            t1 += total[i]
    avew = t1 / t
    return avew
def aveth(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    t = 0
    t1 = 0
    for i in range(len(day)):
        if day[i] == 'Thursday':
            t += 1
            t1 += total[i]
    aveth = t1 / t
    return aveth
def avef(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    t = 0
    t1 = 0
    for i in range(len(day)):
        if day[i] == 'Friday':
            t += 1
            t1 += total[i]
    avef = t1 / t
    return avef
def aves(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    t = 0
    t1 = 0
    for i in range(len(day)):
        if day[i] == 'Saturday':
            t += 1
            t1 += total[i]
    aves = t1 / t
    return aves
def avesun(dataset):
    day = dataset.loc[:,"Day"]
    total = dataset.loc[:, "Total"]
    t = 0
    t1 = 0
    for i in range(len(day)):
        if day[i] == 'Sunday':
            t += 1
            t1 += total[i]
    avesun = t1 / t
    return avesun
print('Monday:', avem(dataset))
print('Tuesday:', avet(dataset))
print('Wednesday:', avew(dataset))
print('Thursday:', aveth(dataset))
print('Friday:', avef(dataset))
print('Saturday:', aves(dataset))
print('Sunday:', avesun(dataset))