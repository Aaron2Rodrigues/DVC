import pandas as pd
import numpy as np 
import sklearn
import joblib
import mlflow
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,recall_score,precision_score
from mlflow.models import infer_signature
import json 
import os 
import mlflow.sklearn
from sklearn.svm import SVC


print("Starting the training process")

data = pd.read_csv("data/iris.csv")
X = data.drop("species" , axis = 1)
y = data["species"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = SVC()  
model.fit(X_train,y_train)

signature = infer_signature(X_train,model.predict(X_train))
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average="macro")
recall = recall_score(y_test,y_pred,average="macro")
precision = precision_score(y_test,y_pred,average="macro")
con_matrix = confusion_matrix(y_test,y_pred)
clas_rep = classification_report(y_test,y_pred)



# Save model using joblib
joblib.dump(model, "models/model.joblib")
model = joblib.load("models/model.joblib")



mlflow.start_run()# run_name = "Iris")
mlflow.log_params(model.get_params())
mlflow.log_metrics({"accuracy-score":accuracy,
                    "f1-score":f1,
                    "recall":recall,
                    "precision":precision})

signature = infer_signature(X_train,model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature,
registered_model_name="Iris Model")









