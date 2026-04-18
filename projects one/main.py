import numpy as  np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data = pd.read_csv('C:/Users/ws/projects one/housing.csv')
df = pd.DataFrame(data)

X = df.drop(['id','date','price'],axis= 1)
y = df['price']

numerical_feauurse = X.columns.tolist()

numericcal_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num',numericcal_transformer,numerical_feauurse)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),

    ('model',RandomForestRegressor(

        n_estimators=100,
        max_depth=10,
        random_state=0

        ))
])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model_pipeline.fit(X_train,y_train)

predictions = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test,predictions)
mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2 * 100:.2f}%')

# নতুন ডেটা তৈরি
new_house_data = pd.DataFrame([{
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2100,
    'sqft_lot': 5000,
    'floors': 2,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'grade': 7,
    'sqft_above': 2100,
    'sqft_basement': 0,
    'yr_built': 2005,
    'yr_renovated': 0,
    'zipcode': "98178",
    'lat': 47.5112,
    'long': -122.257,
    'sqft_living15': 1500,
    'sqft_lot15': 5000
}])

# প্রেডিকশন
price = model_pipeline.predict(new_house_data)
print(f"Predicted Price: ${price[0]:,.2f}")







