import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn

filename = "NYC_Bicycle_Counts_2016_Corrected.csv"
dataset = pd.read_csv(filename)
row = dataset.shape[0]

for i in range(5, 10):
    for j in range(row):
        comma = dataset.iat[j, i].split(",")
        Int = ""
        for x in comma:
            Int += x
        dataset.iat[j, i] = int(Int)

for i in range(2, 5):
    for j in range(row):
        dataset.iat[j, i] = float(dataset.iat[j, i])

def problem1(b1, b2, b3):
    X = dataset[[f"{b1} Bridge", f"{b2} Bridge", f"{b3} Bridge"]].values
    y = dataset[["Total"]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    Predict = model.score(X_test, y_test)
    return model, Predict

Scores = [0, 1, 2, 3]
bmwModel, Scores[0] = problem1("Brooklyn", "Manhattan", "Williamsburg")
bmqModel, Scores[1] = problem1("Brooklyn", "Manhattan", "Queensboro")
bwqModel, Scores[2] = problem1("Brooklyn", "Williamsburg", "Queensboro")
mwqModel, Scores[3] = problem1("Manhattan", "Williamsburg", "Queensboro")

print(f"Total Traffic = {bmwModel.coef_[0][0]}*(Brooklyn Traffic) + {bmwModel.coef_[0][1]}*(Manhattan Traffic) + {bmwModel.coef_[0][2]}*(Williamsburg Traffic) + {bmwModel.intercept_[0]}")
print('')
print("Brooklyn, Manhattan, Queensboro", "r^2 -", Scores[1])
print("Brooklyn, Manhattan, Williamsburg:", "r^2 -", Scores[0])
print("Brooklyn, Williamsburg, Queensboro", "r^2 -", Scores[2])
print("Manhattan, Williamsburg, Queensboro", "r^2 -", Scores[3])

avgList = [0 for i in range(row)]
for i in range(row):
    avgList[i] = (dataset["Total"][i]) / 4

seaborn.distplot(dataset["Brooklyn Bridge"], label="Brooklyn", color = 'red')
seaborn.distplot(dataset["Manhattan Bridge"], label="Manhattan", color = 'pink')
seaborn.distplot(dataset["Williamsburg Bridge"], label="Williamsburg", color = 'blue')
seaborn.distplot(dataset["Queensboro Bridge"], label="Queensboro", color = 'green')
seaborn.distplot(avgList, label="Total", color = 'grey')
plt.legend()
plt.show()

X = dataset[["High Temp", "Low Temp", "Precipitation"]].values
y = dataset[["Total"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=0)
regressor = LinearRegression().fit(X_train, y_train)
col = (dataset[["High Temp", "Low Temp", "Precipitation"]]).columns
coeffs = pd.DataFrame(regressor.coef_, ["coefficients"], columns=col)
y_pred = regressor.predict(X_test)

print('')
print("score: ", regressor.score(X_test, y_test))
print(f"Total Traffic = {regressor.coef_[0][0]}*(High Temp) + {regressor.coef_[0][1]}*(Low Temp) + {regressor.coef_[0][2]}*(Precipitation) + {regressor.intercept_[0]}")
print('')

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