#!/usr/bin/env python
# coding: utf-8

# ### read data from mysql


import pandas as pd
import numpy as np

import MySQLdb
connect =  MySQLdb.connect(host='rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com',
               user= 'root', passwd = 'AI@2019@ai',db= 'stu_db',charset= 'utf8'  )
c = connect.cursor()
c.execute('select * from news_chinese')
rows = c.fetchall()
column = [t[0] for t in c.description]
pd_news_2 = pd.DataFrame(list(rows),columns = column)
pd_news = pd_news_2

# clean the news
import jieba
def cut(string):
    return ' '.join(jieba.cut(string))

import re
def token(string):
    return ' '.join(re.findall(r'[\d|\w]+',string))
  
papers_clean = [cut(token(str(paper))) for paper in pd_news['content'].tolist()]

# turn papers to paper vectors based on tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
vectorized = TfidfVectorizer(max_features=10000)
X = vectorized.fit_transform(papers_clean)
pd_features = pd.DataFrame(X.toarray(),
                           columns = vectorized.get_feature_names())


# model 
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split

# binary the predictor
data_y = pd_news['source']
data_y = [1 if y=='新华社' else 0 for y in data_y]
data_X = pd_features

# split the data into train and test
from sklearn.preprocessing import StandardScaler
def get_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3333)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

rom sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# define the svm model
def model_svm(X_train,y_train):
    svc = SVC(kernel='linear')
    svc.fit(X_train,y_train)
    return svc

# define knn
def model_knn(X_train,y_train):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    return knn

#how to evaluate models
import scipy.misc
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,     precision_score, recall_score, f1_score, cohen_kappa_score,roc_auc_score
list_of_functions = [confusion_matrix,
                     accuracy_score,
                     precision_score,
                     f1_score,
                     recall_score,
                     cohen_kappa_score]

def evalate(result_data,true_data ):
    eva_scores = list(map(lambda x: x(true_data, result_data),
                          list_of_functions))
    return eva_scores

# predict test and evaluate
def predict(out_model,X_test,y_test):
    result_pre = out_model.predict(X_test)
    eva_scores = evalate(result_pre,y_test)
    return eva_scores


#combine all the data, method togethe
def run_model(X,y):
    X_train, X_test, y_train, y_test = get_data(X,y)
    svc = model_svm(X_train,y_train)
    eva_scores = predict(svc,X_test,y_test)
    return eva_scores
  
#run_model(data_X,data_y)
X_train, X_test, y_train, y_test = get_data(data_X,data_y)


svc = model_svm(X_train,y_train)

eva_scores_svm = predict(svc,X_test,y_test)

knn = model_knn(X_train,y_train)

eva_scores_knn = predict(svc,X_test,y_test)

# predict a paper is belongs to 'xinhuashe'

paper_new = '新华社澳门8月24日电（记者胡瑶）澳门特区第五任行政长官选举将于25日上午在澳门东亚运动会体育馆国际会议中心举行，目前选举投票的准备工作已经就绪。

　　根据澳门特区《行政长官选举法》，竞选活动期从选举日前第15日开始至选举日前第2日午夜12时结束。因此，澳门特区第五任行政长官选举的竞选活动在24日进入冷静期，与竞选相关的所有宣传活动当天都已告停。

　　24日上午，澳门特区行政长官选举管理委员会（简称“选管会”）成员到达东亚运动会体育馆国际会议中心，巡视投票站的设置和准备情况，投票站工作人员还就划票、投票、验票、计票等程序进行了演练彩排。

　　澳门特区行政长官崔世安当天中午也来到东亚运动会体育馆国际会议中心，视察选举投票工作的准备情况，并听取选管会汇报。他表示，务必要对选务和相关配套工作检视妥当，保障投票有序、安全和依法完成。

　　选管会主席宋敏莉表示，行政长官选举委员会委员应于25日上午9时至10时，携带个人的澳门永久性居民身份证和投票权证明书行使投票权。整个选举投票过程预计一个半小时内可以完成。

　　根据澳门基本法，澳门特区行政长官由一个具有广泛代表性的选举委员会选出，由中央人民政府任命.'

analysis = vectorized.build_analyzer()
def generate_vector(paper):
    word_list = [cut(token(str(paper)))]
    new_vector = analysis(' '.join(word_list))
    return new_vector

predict(svc,new_vector)

# If predicted reault is '1', the news is from'新华社'. Otherwise, it is not from'新华社'. 
