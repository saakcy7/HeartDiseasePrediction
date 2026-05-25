import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("dataset/heart.csv")
print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
print(data.info())
print(data.duplicated().sum())
data = data.drop_duplicates()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True)  #correlation heatmap
plt.show()

sns.countplot(x='target', data=data)
plt.show()

X = data.drop("target", axis=1)
y = data["target"]

from sklearn.model_selection import train_test_split #train_test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import StandardScaler   #featurescaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score  #accuracy

accuracy = accuracy_score(y_test, rf_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix  #confusion matrix

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm, annot=True)
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, rf_pred))


from sklearn.linear_model import LogisticRegression #logistic regression

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print(accuracy_score(y_test, lr_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, lr_pred))

from sklearn.tree import DecisionTreeClassifier #decisiontree

dt = DecisionTreeClassifier(
 
    max_depth=4,
    min_samples_split=5,
    random_state=42

)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print(accuracy_score(y_test, dt_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, dt_pred))

from sklearn.neighbors import KNeighborsClassifier #knn

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print(accuracy_score(y_test, knn_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, knn_pred))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

rf_scores = cross_val_score(rf_pipeline, X, y, cv=5)

print(rf_scores.mean())





