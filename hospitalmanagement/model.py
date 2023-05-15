import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the time series data into a pandas dataframe
df_train = pd.read_csv(r'C:\Users\Azka\Desktop\FYP\HospitalManagement_Django\hospitalmanagement\PTSD_train - Copy.csv')

# Replacing Missing or Nan values with 0
df_train = df_train.replace(np.NaN, 0)

# print(df_train)
df_train["ptsdpass"].value_counts()

# Seperating data
x=df_train.iloc[:,:-1].values
y=df_train.iloc[:,-1].values

# Resampling data
from imblearn.over_sampling import SMOTE
s=SMOTE()
x_data,y_data=s.fit_resample(x,y)


from collections import Counter
print(Counter(y_data))

# Scaling Data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_scaled=ss.fit_transform(x_data)
# x_scaled

# Dividing into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_data,test_size=0.2,random_state=11)

# Making Logistic Regression model
from sklearn.linear_model import LogisticRegression
l1=LogisticRegression()
history = l1.fit(x_train,y_train)

# Assigning Prediction Function
y_pred=l1.predict(x_test)
print(y_pred)

#y_test

# Calculating Accuracy Score
from sklearn.metrics import accuracy_score
ab=accuracy_score(y_test,y_pred)*100
print(ab)


# saving model as a pickle
import pickle
pickle.dump(l1,open("ml_model.sav", "wb"))
pickle.dump(ss, open("scaler.sav", "wb"))