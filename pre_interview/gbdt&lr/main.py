#!/usr/bin/python
# -*- coding: UTF-8 -*-
#from han import HAN

import os
import data_preprocess
import tensorflow as tf
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from collections import Counter


def save_model(model,model_path):
    joblib.dump(model, model_path)    
    return 

def load_model(model_path):
    return joblib.load(model_path)

def write_best_param(file_path,model_name,param_dict):
    with open(file_path,'a') as writer:
        writer.write(model_name+':\n')
        for key in param_dict.keys():
            writer.write(key+':'+str(param_dict[key])+'\n')
        writer.write(model_name+' finish'+'\n'+'\n') 

def logistic_regression_training(model_path,x_train,y_train,if_adujst=False,params=None):
    if if_adujst:
        if __name__=='__main__':
            LR = LogisticRegression()
            clf = GridSearchCV(LR,params,cv = 5,scoring = 'accuracy',verbose=1)
            clf.fit(x_train, y_train)
            best_estimator = clf.best_estimator_
            print(clf.best_score_)
            print(clf.best_params_)
            save_model(best_estimator, model_path)
    else:  
        clf = LogisticRegressionCV(Cs=5,cv=5,solver='sag',max_iter=500)
        clf.fit(x_train,y_train)
        save_model(clf, model_path)
        

def gbdt_training(model_path,x_train,y_train,if_adujst=False,params=None):
    if if_adujst:
        if __name__=='__main__':
            gbdt = GradientBoostingClassifier(random_state=7)
            #clf = GridSearchCV(gbdt,params,scoring = 'roc_auc',n_jobs=5,verbose=1)
            clf = RandomizedSearchCV(gbdt,params,scoring = 'accuracy',verbose=1)
            clf.fit(x_train, y_train)
            best_estimator = clf.best_estimator_
            print(clf.best_score_)
            print(clf.best_params_)
            #write_best_param(best_param_record, 'gbdt', clf.best_params_)
            save_model(best_estimator, model_path)
    else:  
        clf = GradientBoostingClassifier(learning_rate=0.1,n_estimators=60,max_features='sqrt',subsample=0.8,random_state=7,
                                         min_samples_leaf=50,min_samples_split=500,max_depth=8)
        clf.fit(x_train,y_train)
        save_model(clf, model_path)


if __name__ == '__main__':
    vocab_file = ''
    padding_size = 200
    data_file = ''
    feature_list = []
    class_list = []
    label_name = ""

    X_train, X_test, Y_train, Y_test,label2id = data_preprocess.set_dataset_sif(data_file, vocab_file, feature_list, class_list, label_name, padding_size)

    #sif embedding
    w2vfile = 'sif/data/medical_token_vec_100.bin' # word vector file, can be downloaded from GloVe website
    weightfile = 'sif/auxiliary_data/medical_character_vocab_freq.txt' # each line is a word and its frequency
    weightpara = 3e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    
    #进行embedding的目标必须是分词完成后的语料（字符串类型，空格隔开每个词）
    X_train_sif_embedding = data_preprocess.get_sif_embedding(w2vfile, weightfile, weightpara,X_train)
    X_test_sif_embedding = data_preprocess.get_sif_embedding(w2vfile, weightfile, weightpara, X_test)
    #训练步骤
    logistic_model_adjust=''
    best_param_record=''
    gbdt_model_adjust=''
    #逻辑回归
    
    params_lr = dict(C=[0.001, 0.01, 0.1, 1, 10, 100, 1000],penalty=['l2'])
    
    #gbdt
    params_gbdt = dict(n_estimators=range(20,81,10),max_depth=range(5,16,2), min_samples_split=range(200,1001,200),
                           min_samples_leaf=range(30,71,10),max_features=range(7,20,2))

    lr_model = joblib.load(logistic_model_adjust)
    gbdt_model = joblib.load(gbdt_model_adjust)

    lr_y_predict_test = lr_model.predict(X_test_sif_embedding)
    gbdt_y_predict_test = gbdt_model.predict(X_test_sif_embedding)

    lr_y_score = lr_model.decision_function(X_test_sif_embedding)
    gbdt_y_score = gbdt_model.decision_function(X_test_sif_embedding)

    print('testing sets accuracy for lr:')
    print(accuracy_score(y_true=Y_test, y_pred=lr_y_predict_test))
    print('testing sets confusion matrix for lr:')
    print(classification_report(Y_test, lr_y_predict_test))

    print('testing sets accuracy for gbdt:')
    print(accuracy_score(y_true=Y_test, y_pred=gbdt_y_predict_test))
    print('testing sets confusion matrix for lr:')
    print(classification_report(Y_test, gbdt_y_predict_test))
