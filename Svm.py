# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:24:59 2020

@author: Mete


Svm:support vector machine
Bir siniflandirma biçimidir.Basit olarak düşünürsek belirli noktalar
kümesi etrafındaki dataları birbirinden fit etmek için kullandığımız
bir algoritma biçimidir

"""

#%% library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% read csv
data=pd.read_csv("svmdataset.csv")
#%% gereksiz sütunları siliyoruz
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


#%%iyi ve kötü huylu tumorleri bir birinden ayırt etmek için
#doku ve yarıçapına göre görsel grafik oluşturuyoruz

M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]
#scatter plot alpha=saydamlık
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kotu",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


#%%string olan sonuç sütünlarını int değerlere çeviriyoruz
#Kötü huylu "M" olanı 1 iyi ise 0 yapıyoruz
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#%% svm
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(x_train,y_train)


#%%x test


print("accuracy o f svm algo:",svm.score(x_test,y_test))




    






