import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,classification_report,recall_score,precision_score,accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
#DataFrame
df=pd.read_csv('drug200.csv')  #loading into pandas DataFrame
df
Age	Sex	BP	Cholesterol	Na_to_K	Drug
0	23	F	HIGH	HIGH	25.355	DrugY
1	47	M	LOW	HIGH	13.093	drugC
2	47	M	LOW	HIGH	10.114	drugC
3	28	F	NORMAL	HIGH	7.798	drugX
4	61	F	LOW	HIGH	18.043	DrugY
...	...	...	...	...	...	...
195	56	F	LOW	HIGH	11.567	drugC
196	16	M	LOW	HIGH	12.006	drugC
197	52	M	NORMAL	HIGH	9.894	drugX
198	23	M	NORMAL	NORMAL	14.020	drugX
199	40	F	LOW	NORMAL	11.349	drugX
200 rows × 6 columns

df.isnull().sum()
df.info()
df['Age']=pd.cut(df['Age'],bins=[15,30,50,75],labels=['young','adult','old'])
​
df
Age	Sex	BP	Cholesterol	Na_to_K	Drug
0	young	F	HIGH	HIGH	25.355	DrugY
1	adult	M	LOW	HIGH	13.093	drugC
2	adult	M	LOW	HIGH	10.114	drugC
3	young	F	NORMAL	HIGH	7.798	drugX
4	old	F	LOW	HIGH	18.043	DrugY
...	...	...	...	...	...	...
195	old	F	LOW	HIGH	11.567	drugC
196	young	M	LOW	HIGH	12.006	drugC
197	old	M	NORMAL	HIGH	9.894	drugX
198	young	M	NORMAL	NORMAL	14.020	drugX
199	adult	F	LOW	NORMAL	11.349	drugX
200 rows × 6 columns

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Age']=df['Age'].astype('str')
df['Sex']=df[['Sex']].apply(le.fit_transform)
df['Age']=df[['Age']].apply(le.fit_transform)
df['BP']=df[['BP']].apply(le.fit_transform)
df['Cholesterol']=df[['Cholesterol']].apply(le.fit_transform)
df['Drug']=df[['Drug']].apply(le.fit_transform)
.shape
df.shape
(200, 6)
df['Drug'].value_counts()
mean = 16.0844
standard_deviation = 7.2
​
x_values = df['Na_to_K']
y_values = scipy.stats.norm(mean, standard_deviation)
​
plt.plot(x_values, y_values.pdf(x_values))

plt.hist(df['Na_to_K'],bins=9)
(array([35., 59., 35., 25., 13., 15.,  8.,  6.,  4.]),
 array([ 6.269     ,  9.82211111, 13.37522222, 16.92833333, 20.48144444,
        24.03455556, 27.58766667, 31.14077778, 34.69388889, 38.247     ]),


#finding the correlation
df.corr()

 
#plotting the correlation using heatmap
 
 
sns.heatmap(data=df.corr(),annot=True)


x=df.drop('Drug',axis=1)
y=df['Drug']

#feature selection
f=SelectKBest(score_func=chi2,k=5)
 
anova=SelectKBest(score_func=f_classif,k=5)
 
an=anova.fit(x,y)
ff=f.fit(x,y)
ff.scores_
array([ 10.76681017,   1.01723924,  70.16761261,  10.09897371,
       411.38934808])
 
an.scores_
array([ 2.64836529,  0.52209909, 44.13614444,  5.29945672, 85.6113565 ])
 
#train test split
#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
 

#algorithm- Loigstic regression
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(recall_score(y_test,pred,average='macro'))
print(precision_score(y_test,pred,average='macro'))
print(accuracy_score(y_test,pred))
            
0.85
 
cv=StratifiedKFold(n_splits=6,shuffle=False,random_state=1)
n_scores=cross_val_score(lr,x_train,y_train,cv=cv,n_jobs=-1)
print(mean(n_scores),std(n_scores))
 
cv1=KFold(n_splits=6,shuffle=False,random_state=1)
kfoldscore=cross_val_score(lr,x_train,y_train,cv=cv1,n_jobs=-1)
np.mean(n_scores)
 
0.8931159420289855
 
print(kfoldscore)
print(np.mean(kfoldscore))

[0.875      0.83333333 0.95652174 1.         0.86956522 0.82608696]
0.8934178743961354
 
 
 #algorithm - Decisiontreeclassifier
 
 
treee=DecisionTreeClassifier()
treee.fit(x_train,y_train)
treepred=treee.predict(x_test)
print(classification_report(y_test,treepred))
print(confusion_matrix(y_test,treepred))
print(recall_score(y_test,treepred,average='micro'))
print(precision_score(y_test,treepred,average='weighted'))
print(accuracy_score(y_test,treepred))
                     
1.0

 
 #algorithm - Randomforest
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
rfpred=rf.predict(x_test)
print(classification_report(y_test,rfpred))
print(confusion_matrix(y_test,rfpred))
print(recall_score(y_test,rfpred,average='micro'))
print(precision_score(y_test,rfpred,average='weighted'))
print(accuracy_score(y_test,rfpred))
            
1.0
1.0

