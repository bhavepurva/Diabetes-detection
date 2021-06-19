import zipfile
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score, recall_score,roc_auc_score, accuracy_score

size_of_test_set=0.2
shuffle=True

with zipfile.ZipFile("diab.zip","r") as zip_ref:
  zip_ref.extractall("")

#getting the data
data=pd.read_csv("diabetes.csv")

#features and labels
y=data["Outcome"]
x=data.drop("Outcome",axis=1)

#determine null values or missing values
age=x["Age"]
preg=x["Pregnancies"]
x=x[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction"]].replace(0,np.nan)

#imputer to impute mean value for each feature
imputer=SimpleImputer(strategy="mean")
imputer.fit(x)

#transforming feature data
X=imputer.transform(x)
x_transf=pd.DataFrame(X,columns=x.columns,index=x.index)
x_transf["Pregnancies"]=preg
x_transf["Age"]=age

#scaling values using minmaxscaler in range 0 to 1
scale=MinMaxScaler()
x_transf[x_transf.columns] = pd.DataFrame(scale.fit_transform(x_transf[x_transf.columns].values), columns=x_transf.columns, index=x_transf.index)
x_transf.head()

#split data into training and testing set, maintaining same ratio of outcome
x_train,x_test=train_test_split(x_transf,test_size=size_of_test_set,random_state=42,shuffle=shuffle)
y_train,y_test=train_test_split(y,test_size=size_of_test_set,random_state=42,shuffle=shuffle)

#nn
#the model saved before can be now directly loaded witout having to train it again
neural_model=keras.models.load_model("my_model.h5")
neural_model.summary()
acc_param=neural_model.evaluate(x_test,y_test)
print(acc_param)

#sgd
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(x_train, y_train)

y_test_pred=sgd_clf.predict(x_test)
print("Testing accuracy for sgd: "+str(accuracy_score(y_test, y_test_pred)))

cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)
print("Training precision for sgd: "+str(precision_score(y_train, y_train_pred)))
print("Training recall for sgd: "+str(recall_score(y_train, y_train_pred)))

cross_val_score(sgd_clf, x_test, y_test, cv=3, scoring="accuracy")
y_test_pred = cross_val_predict(sgd_clf, x_test, y_test, cv=3)
confusion_matrix(y_test, y_test_pred)
print("Testing precision for sgd: "+str(precision_score(y_test, y_test_pred)))
print("Testing recall for sgd: "+str(recall_score(y_test, y_test_pred)))

y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=3,method="decision_function")
print("Training auc for sgd: "+str(roc_auc_score(y_train, y_scores)))

y_scores = cross_val_predict(sgd_clf, x_test, y_test, cv=3,method="decision_function")
print("Testing auc for sgd: "+str(roc_auc_score(y_test, y_scores)))

#knn
knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)
knn_clf.fit(x_train, y_train)

y_test_pred=knn_clf.predict(x_test)
print("Testing accuracy for knn: "+str(accuracy_score(y_test, y_test_pred)))

cross_val_score(knn_clf, x_train, y_train, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(knn_clf, x_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)
print("Training precision for knn: "+str(precision_score(y_train, y_train_pred)))
print("Training recall for knn: "+str(recall_score(y_train, y_train_pred)))

cross_val_score(knn_clf, x_test, y_test, cv=3, scoring="accuracy")
y_test_pred = cross_val_predict(knn_clf, x_test, y_test, cv=3)
confusion_matrix(y_test, y_test_pred)
print("Testing precision for knn: "+str(precision_score(y_test, y_test_pred)))
print("Testing recall for knn: "+str(recall_score(y_test, y_test_pred)))

y_scores = cross_val_predict(knn_clf, x_train, y_train, cv=3)
print("Training auc for knn: "+str(roc_auc_score(y_train, y_scores)))

y_scores = cross_val_predict(knn_clf, x_test, y_test, cv=3)
print("Testing auc for knn: "+str(roc_auc_score(y_test, y_scores)))
