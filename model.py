import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Dataset.csv")
X=df[["Average Cost for two", "Has Table booking", "Has Online delivery", "Price range"]]
Y=df["Aggregate rating"]
from sklearn.preprocessing import LabelEncoder
LabelEncoder=LabelEncoder()
X["Has Table booking"]=LabelEncoder.fit_transform(X["Has Table booking"])                                                                      
X["Has Online delivery"]=LabelEncoder.fit_transform(X["Has Online delivery"])
from sklearn.preprocessing import StandardScaler    
scaler = StandardScaler()
x=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, )
def prediction(result):
    print("Mean Absolute Error: ", mean_absolute_error(y_test, result))
    print("Mean Squared Error: ", mean_squared_error(y_test, result))
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train, y_train)
prediction_model = model.predict(x_test)
prediction (prediction_model)
from sklearn.model_selection import GridSearchCV
param_grid ={
    'fit_intercept': [True, False],
    'positive': [True, False],
    'copy_X': [True, False],
    'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
model=LinearRegression()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)

print ("best parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
mae= mean_absolute_error(y_test, y_pred)
mse= mean_squared_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
coefs = np.abs(best_model.coef_)
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': coefs})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
import pickle
with open('rating_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
