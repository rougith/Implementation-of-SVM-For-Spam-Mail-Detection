# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and load the dataset.
2. Separate input messages and output labels.
3. Split data into training and testing sets.
4. Convert text into numerical form and train the SVM model.
5. Predict results and evaluate accuracy.
 
  

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rougith D
RegisterNumber: 212225220087
*/

import pandas as pd
data=pd.read_csv(r"C:\Users\acer\Downloads\spam.csv",encoding='latin -1') 
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
Data
<img width="1125" height="612" alt="Screenshot 2026-05-20 160741" src="https://github.com/user-attachments/assets/6ffe7faa-57af-4f3c-9e20-f2861cef7cb2" />

Accuracy
<img width="1123" height="92" alt="Screenshot 2026-05-20 160753" src="https://github.com/user-attachments/assets/d506b9d1-aa6c-43bc-839c-333d72796283" />

confusion matrix


<img width="627" height="97" alt="Screenshot 2026-05-20 160758" src="https://github.com/user-attachments/assets/c0d2444c-3447-4fa9-85b9-8d64c5ac69fb" />


Report
<img width="1122" height="360" alt="Screenshot 2026-05-20 160807" src="https://github.com/user-attachments/assets/95577717-dae2-4eba-ada6-f38f1f9b49cc" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
