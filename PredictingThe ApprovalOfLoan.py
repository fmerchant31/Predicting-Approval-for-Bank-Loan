#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter("ignore")

df=pd.read_csv("11-2-Dataset-Predicting Approval for Bank Loan.csv")
df.head()
#url = "https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
#n= ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
#df=pd.read_csv(url,names=n)

#df.head()


# In[121]:


df=df.drop(["Loan_ID","Gender","Married"],axis=1)


# In[122]:


df.head()


# In[123]:


df.isnull().sum()


# In[124]:


df["Dependents"]= df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"]= df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["LoanAmount"]= df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"]= df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"]= df["Credit_History"].fillna(df["Credit_History"].median())
#x["Loan_Status"]= x["Loan_Status"].fillna(x["Loan_Status"].mode()[0])


# In[125]:


df.isnull().sum()


# In[126]:


df.describe()


# In[127]:


df.shape


# In[128]:


from sklearn.preprocessing import LabelEncoder
categorical = ['Dependents','Education','Self_Employed','Property_Area','Loan_Status']
Encoder = LabelEncoder() 
for col in categorical :
    df[col]= Encoder.fit_transform(df[col])
df.head()


# In[129]:


x=df.drop('Loan_Status',axis=1)
y=df['Loan_Status']


# In[130]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[131]:


from sklearn.linear_model import LogisticRegression
my_model= LogisticRegression()
result = my_model.fit(x_train, y_train)


# In[132]:


predictions= result.predict(x_test)
predictions


# In[133]:


print("\n**Classification Report of LogisticRegression:\n",metrics.classification_report(y_test,predictions))


# In[134]:


print("accuracy of LogisticRegression:",accuracy_score(y_test, predictions))


# In[135]:


con =confusion_matrix(predictions,y_test)
conf = pd.DataFrame(con, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
print("Confusion metrix for LogisticRegression:")
conf


# In[136]:


#from sklearn.linear_model import LinearRegression
#my_model = LinearRegression()  
#re0 = my_model.fit(x_train, y_train)
#pr0 = re0.predict(x_test)
#print("LinearRegression:",pr0)

from sklearn.linear_model import LinearRegression
my_model = LinearRegression()  
re0 = my_model.fit(x_train, y_train)
pr0 = re0.predict(x_test)
print("LinearRegression:",pr0)


# In[137]:



from sklearn.metrics import r2_score
a=r2_score(y_test,pr0)
print("Accuracy of LinearRegression:",a)


# In[138]:


#plt.scatter(x_train, y_train, color ='c') 
#plt.plot(x_test, pr0, color ='b') 


# In[139]:


from sklearn.neighbors import KNeighborsClassifier
my_model = KNeighborsClassifier(n_neighbors = 3)
re1 = my_model.fit(x_train,y_train)
pr1 = re1.predict(x_test)
print("KNeighbors:",pr1)


# In[140]:


print("With KNN (K=3) accuracy is:" , re1.score(x_test,y_test))


# In[141]:


con =confusion_matrix(pr1,y_test)
conf = pd.DataFrame(con, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
print("Confusion metrix for KNeighbors:")
conf


# In[142]:


print('\n**Classification Report of KNN:\n',metrics.classification_report(y_test,pr1))


# In[143]:


from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
re2=my_model.fit(x_train, y_train)
pr2 = re2.predict(x_test)
print("RandomForestClassifier:",pr2)


# In[144]:


print("Accuracy of RandomForestClassifier:",metrics.accuracy_score(y_test, pr2))


# In[145]:


con =confusion_matrix(pr2,y_test)
conf = pd.DataFrame(con, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
print("Confusion metrix for RandonForestTree:")
conf


# In[146]:


print('\n**Classification Report of RandomForest:\n',metrics.classification_report(y_test,pr2))


# In[147]:


from sklearn.tree import DecisionTreeClassifier
m=DecisionTreeClassifier(random_state=0)
x_train=x_train.astype('int')
y_train=y_train.astype('int')
re3=m.fit(x_train,y_train)
pr3=re3.predict(x_test)
print("DecisionTreeClassifier:",pr3)


# In[148]:


print("Accuracy of DecisionTreeClassifier:",metrics.accuracy_score(y_test, pr3))


# In[149]:


con =confusion_matrix(pr3,y_test)
conf = pd.DataFrame(con, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
print("Confusion metrix for DecissionTree:")
conf


# In[150]:


print('\n**Classification Report of Decission tree:\n',metrics.classification_report(y_test,pr3))


# In[151]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # sc is an object of StandardScaler.
df_scaled = sc.fit_transform(df)


# In[152]:


from sklearn.cluster import KMeans
ssq =[]
for K in range(1,11): # 1 to 10
    my_model = KMeans(n_clusters=K,random_state=123)
re4 = my_model.fit(df_scaled)
ssq.append(my_model.inertia_)
my_model = KMeans(n_clusters=3, random_state=123)
re5= my_model.fit(df_scaled)
pr5=re5.predict(df_scaled)
print("KMeans",pr5)


# In[ ]:





# In[ ]:





# In[153]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
x_tr=sc.fit_transform(x_train)
x_ts=sc.transform(x_test)
from sklearn.svm import SVC
m=SVC(kernel='rbf',random_state=123)
re6=m.fit(x_train,y_train)
pr6=re6.predict(x_test)
print("SVM:",pr6)


# In[154]:


print("accuracy of SVM :",metrics.accuracy_score(y_test,pr6))


# In[155]:


import seaborn as sn
from sklearn.metrics import confusion_matrix
con =confusion_matrix(pr6,y_test)
conf = pd.DataFrame(con, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
conf



# In[156]:


print('\n**Classification Report of SVM:\n',metrics.classification_report(y_test,pr6))


# In[163]:


import seaborn as sns
sns.countplot(x= 'Loan_Status',data=df)


# In[164]:


sns.boxplot(x='Self_Employed',y='Loan_Status', data=df)


# In[159]:



sns.violinplot(x='ApplicantIncome',y='LoanAmount',data=df)


# In[160]:


plt.title("Histogram")
plt.xlabel("LoanAmount")
plt.ylabel("Frequency")
plt.hist(df["LoanAmount"],100)
plt.show()


# In[105]:


#accuracy of LogisticRegression: 0.8373983739837398
#Accuracy of LinearRegression: 0.25774254506788297
#Accuracy of DecisionTreeClassifier: 0.6666666666666666
#Accuracy of RandomForestClassifier: 0.8130081300813008
#With KNN (K=3) accuracy is: 0.6016260162601627
#accuracy of SVM : 0.7317073170731707
print("Maximum Accuracy is of LogisticsRegression=0.827298373983")
print("So, we conclude that best amodel is alogiticaregession")


# In[ ]:




