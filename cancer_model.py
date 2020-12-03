import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data=pd.read_csv("cancer.csv")
data.drop(["Unnamed: 32","id"],axis="columns",inplace=True)
res=pd.get_dummies(data["diagnosis"])
cancer=pd.concat([data,res],axis="columns")
cancer.drop(["diagnosis","B"],axis="columns",inplace=True)
cancer.rename(columns={"M":"Malignant/Benign"},inplace=True)

x=cancer.drop(["Malignant/Benign"],axis="columns")
y=cancer["Malignant/Benign"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=20,random_state=13)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test, y_pred)

pickle.dump(model,open("cancer_model.pkl","wb"))
