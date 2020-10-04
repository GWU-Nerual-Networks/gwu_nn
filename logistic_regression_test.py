import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense
from gwu_nn.activation_layers import Sigmoid, RELU


y_col = 'Survived'
x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = pd.read_csv('titanic_data.csv')
y = np.array(df[y_col]).reshape(-1, 1)
orig_X = df[x_cols]

# Lets standardize our features
scaler = preprocessing.StandardScaler()
stand_X = scaler.fit_transform(orig_X)
X = stand_X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

network = GWUNetwork()
network.add(Dense(6, 14, True, activation='relu'))
network.add(Dense(14, 1, True, activation='sigmoid'))
network.compile(loss='log_loss', lr=.5)
network.fit(X_train, y_train, batch_size=10, epochs=100)
