import pandas as pd
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

mapping = {
    'A': 0,
    'C': 1,
    'T': 2,
    'G': 3,
}

df = pd.read_csv('data.csv')
X = df['seq']
X = [[mapping[sym] for sym in line][:1000] for line in X]
X = np.array(X, dtype=np.float32)

y = df['class'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)


model = catboost.CatBoostClassifier(iterations=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)

print(f1_score(y_valid, y_pred))