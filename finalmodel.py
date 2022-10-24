import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv('features.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
xgb = XGBClassifier(n_estimators=200,
                    min_samples_split=4,
                    min_impurity_decrease=0.0,
                    max_features='log2',
                    max_depth=10,
                    learning_rate=0.5)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print(accuracy_score(y_test, y_pred))

xgb.save_model('modelxgb.json')
