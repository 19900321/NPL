#!/usr/bin/env python
# coding: utf-8

# ### read data from mysql

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import MySQLdb
connect =  MySQLdb.connect(host='rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com',
               user= 'root', passwd = 'AI@2019@ai',db= 'stu_db',charset= 'utf8'  )
c = connect.cursor()
c.execute('select * from news_chinese')
rows = c.fetchall()
column = [t[0] for t in c.description]
pd_news_2 = pd.DataFrame(list(rows),columns = column)


# In[3]:


pd_news = pd_news_2


# ### clean the news

# In[4]:


import jieba
def cut(string):
    return ' '.join(jieba.cut(string))


# In[5]:


import re
def token(string):
    return ' '.join(re.findall(r'[\d|\w]+',string))


# In[6]:


papers_clean = [cut(token(str(paper))) for paper in pd_news['content'].tolist()]


# ### turn papers to paper vectors based on tfidf

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


vectorized = TfidfVectorizer(max_features=10000)


# In[9]:


X = vectorized.fit_transform(papers_clean)


# In[10]:


pd_features = pd.DataFrame(X.toarray(),
                           columns = vectorized.get_feature_names())


# ### model 

# In[11]:


from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split


# In[12]:


# binary the predictor
data_y = pd_news['source']
data_y = [1 if y=='新华社' else 0 for y in data_y]


# In[13]:


data_X = pd_features


# In[25]:


# split the data into train and test
from sklearn.preprocessing import StandardScaler
def get_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3333)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# In[15]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[16]:


# define the svm model
def model_svm(X_train,y_train):
    svc = SVC(kernel='linear')
    svc.fit(X_train,y_train)
    return svc


# In[18]:


# define knn
def model_knn(X_train,y_train):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    return knn


# In[19]:


#how to evaluate models
import scipy.misc
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,     precision_score, recall_score, f1_score, cohen_kappa_score,roc_auc_score
list_of_functions = [confusion_matrix,
                     accuracy_score,
                     precision_score,
                     f1_score,
                     recall_score,
                     cohen_kappa_score]


# In[20]:


def evalate(result_data,true_data ):
    eva_scores = list(map(lambda x: x(true_data, result_data),
                          list_of_functions))
    return eva_scores


# In[21]:


# predict test and evaluate
def predict(out_model,X_test,y_test):
    result_pre = out_model.predict(X_test)
    eva_scores = evalate(result_pre,y_test)
    return eva_scores


# In[23]:


#combine all the data, method togethe
def run_model(X,y):
    X_train, X_test, y_train, y_test = get_data(X,y)
    svc = model_svm(X_train,y_train)
    eva_scores = predict(svc,X_test,y_test)
    return eva_scores


# In[134]:


#run_model(data_X,data_y)


# In[26]:


X_train, X_test, y_train, y_test = get_data(data_X,data_y)


# In[ ]:


svc = model_svm(X_train,y_train)


# In[ ]:


eva_scores_svm = predict(svc,X_test,y_test)


# In[ ]:


knn = model_knn(X_train,y_train)


# In[ ]:


eva_scores_knn = predict(svc,X_test,y_test)


# ###  predict a paper is belongs to 'xinhuashe'

# In[ ]:


paper_new = ''


# In[ ]:


analysis = vectorized.build_analyzer()
def generate_vector(paper):
    word_list = [cut(token(str(paper)))]
    new_vector = analysis(' '.join(word_list))
    return new_vector


# In[ ]:


predict(svc,new_vector)


# If predicted reault is '1', the news is from'新华社'. Otherwise, it is not from'新华社'. 
