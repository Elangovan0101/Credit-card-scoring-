import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("train.csv")

data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, "Good": 2, "Bad": 0})

x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
"Num_Bank_Accounts", "Num_Credit_Card", 
"Interest_Rate", "Num_of_Loan", 
"Delay_from_due_date", "Num_of_Delayed_Payment", 
"Credit_Mix", "Outstanding_Debt", 
"Credit_History_Age", "Monthly_Balance"]])
y = np.array(data["Credit_Score"])  

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

def predict_credit_score(a, b, c, d, e, f, g, h, i, j, k, l):
    features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
    return model.predict(features)

print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

predicted_score = predict_credit_score(a, b, c, d, e, f, g, h, i, j, k, l)
print("Predicted Credit Score =", predicted_score)
