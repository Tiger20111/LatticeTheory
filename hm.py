import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/tiger/tmp/output.csv', header=0)

df.drop(df.columns[[0, 4, 5, 9, 10, 11]],
   axis = 1, inplace = True)

y = df['rarity']
X = df.drop(columns=['rarity'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=23)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

df.head()
