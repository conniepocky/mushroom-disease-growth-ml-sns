import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras 

# Load the data

data = pd.read_csv('data/disease_growth_level.csv')

data['disease growth possibility level'] = data['disease growth possibility level'].map(lambda x: 2 if x == 'High' else 1 if x == 'Moderate' else 0)
data["ventilation"] = data["ventilation"].map(lambda x: 2 if x == 'high' else 1 if x == 'medium' else 0)
data["light_intensity"] = data["light_intensity"].map(lambda x: 2 if x == 'high' else 1 if x == 'medium' else 0)
data = data.drop(["date", "time"], axis=1)

for x in data.columns:
    if x != 'disease growth possibility level' or "ph":
        data[x] = data[x].fillna(data[x].mode())

data.ph = data.ph.fillna(data.ph.mean())

print(data.head())

print(data.describe())

x = data.drop("disease growth possibility level", axis=1)
y = data["disease growth possibility level"]

# Data visualization

sns.barplot(x='ph',y='disease growth possibility level',data=data, palette="crest")
plt.xticks(rotation=90)
sns.relplot(x="temperature", y="disease growth possibility level", data=data)
plt.show()

# predictions

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)

from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(y_test, pred.round()))
