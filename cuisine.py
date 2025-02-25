import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cuisine_data = pd.read_csv('Dataset.csv')
cuisine_data.head()
drop_cols = ['Restaurant ID','Restaurant Name','Address','Locality','Locality Verbose','Currency','Rating color','Rating text','votes']
cuisine_data = cuisine_data.drop(columns=[col for col in cuisine_data.columns if col in drop_cols])
cuisine_data.head()
cuisine_data = cuisine_data.dropna(subset=['Cuisines'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cuisine_data['Has Table booking'] = le.fit_transform(cuisine_data['Has Table booking'])
cuisine_data['Has Online delivery'] = le.fit_transform(cuisine_data['Has Online delivery'])
cuisine_data['Is delivering now'] = le.fit_transform(cuisine_data['Is delivering now'])
cuisine_data['Switch to order menu'] = le.fit_transform(cuisine_data['Switch to order menu'])
cuisine_data['City'] = le.fit_transform(cuisine_data['City'])
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(cuisine_data['Cuisines'])
cuisine_data = cuisine_data.drop(columns=['Cuisines'])
scaler = StandardScaler()
X = scaler.fit_transform(cuisine_data)
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y.sum(axis=1))
x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
params = {'estimator__n_estimators': [100], 'estimator__max_depth': [10,20, None ], 'estimator__min_samples_split': [2,5,10]}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_ovr = OneVsRestClassifier(rf)
rf_ovr.fit(x_train, y_train)
y_pred = rf_ovr.predict(x_test)
print(f1_score(y_test, y_pred, average='micro'))
import pickle
with open('cuisine_model.pkl', 'wb') as file:
    pickle.dump(rf_ovr, file)

