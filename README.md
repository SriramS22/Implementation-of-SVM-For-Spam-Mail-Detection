# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sriram.S
RegisterNumber:  212222240105
*/

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()



x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)


y_pred = svc.predict(x_test)
y_pred



from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

# Result
![image](https://github.com/SriramS22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119094390/ca141080-147b-4955-85c7-8ba012e40562)

# data.head()
![image](https://github.com/SriramS22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119094390/8ccdbea8-0401-4083-9b5b-8f587c2e8820)

# data.info()
![image](https://github.com/SriramS22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119094390/35a369a3-c068-4230-a320-96e8063cc183)

# data.isnull().sum()
![image](https://github.com/SriramS22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119094390/60625367-f508-411f-999c-de26f0e7e4eb)

# Accuracy value
![image](https://github.com/SriramS22/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119094390/d4452ff5-839b-489a-8f9a-a9ebc8aeea00)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
