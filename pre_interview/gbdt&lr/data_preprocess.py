#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import pandas as pd
from tensorflow.keras import preprocessing
import numpy as np
import json
from sklearn.model_selection import train_test_split
import pkuseg
import gensim
import sys

sys.path.append('sif/src')
import data_io, params, SIF_embedding
from data_utils import read_samples
import random


def set_dataset_sif(data_file, vocab_file, feature_list: list, class_list: list, label_name, padding_size):
    print("Loading data from {} ...".format(data_file))
    # seg = pkuseg.pkuseg()
    df = pd.read_csv(data_file)
    df.dropna(subset=['CURRENT_MEDICAL_HISTORY', 'PHYSICAL_EXAMINATION'], inplace=True)
    label2id = {item: class_list.index(item) for item in class_list}
    new_train_x_pd = df[feature_list]
    new_train_y_pd = df[label_name]

    new_train_x_pd['sex'] = new_train_x_pd['Sex_Code'].apply(sexcode2sex)
    sex = new_train_x_pd['sex'].values
    age = new_train_x_pd['age'].values
    complaint = new_train_x_pd["COMPLAINT"].values
    current_medical_history = new_train_x_pd["CURRENT_MEDICAL_HISTORY"].values
    previous_history = new_train_x_pd["PREVIOUS_HISTORY"].values
    allergy_history = new_train_x_pd["ALLERGY_HISTORY"].values
    family_history = new_train_x_pd["FAMILY_HISTORY"].values
    # 增加体格检查信息
    # physical_examination = new_train_x_pd["PHYSICAL_EXAMINATION"].values

    new_x = ["性别:" + str(i[0]) + "年龄:" + str(i[1]) + "岁，" + "主诉:" + str(i[2]) + " 病症:" + str(i[3]) + " 病史:" + str(
        i[4]) + "过敏史:" + str(i[5]) + "家族史:" + str(i[6]) for i in
             zip(sex, age, complaint, current_medical_history, previous_history, allergy_history, family_history)]
    new_y = new_train_y_pd.values
    y = [label2id[item] for item in list(new_y)]
    X_train, X_test, Y_train, Y_test = train_test_split(new_x, y, shuffle=True, random_state=1)
    samples, labels = read_samples('data/embed_replaced.txt')
    X_train += samples
    Y_train += [2 if label == '|2' else int(label) for label in labels]
    X_train = [' '.join(list(sentence)) for sentence in X_train]
    X_test = [' '.join(list(sentence)) for sentence in X_test]
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(X_train)
    random.seed(randnum)
    random.shuffle(Y_train)
    return X_train, X_test, Y_train, Y_test, label2id


def get_sif_embedding(wordfile, weightfile, weightpara, source_text):
    rmpc = 1  # number of principal components to remove in SIF weighting scheme
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    # word2weight['str'] is the weight for the word 'str'
    word2weight = data_io.getWordWeight(weightfile, weightpara)
    # weight4ind[i] is the weight for the i-th word
    weight4ind = data_io.getWeight(words, word2weight)
    # load sentences for medical_record
    # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    x_mr, m_mr = data_io.sentences2idx(source_text, words)
    # get word weights
    w_mr = data_io.seq2weight(x_mr, m_mr, weight4ind)
    # set parameters
    sif_params = params.params()
    sif_params.rmpc = rmpc
    # get SIF embedding for medical_record
    # embedding[i,:] is the embedding for sentence i
    embedding_mr = SIF_embedding.SIF_embedding(We, x_mr, w_mr, sif_params)
    return embedding_mr
