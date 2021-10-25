#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from data_utils import read_samples, isChinese, write_samples
from itertools import islice
import pkuseg


def extract_keywords(dct, tfidf, threshold=0.2, topk=5):
    """find high TFIDF socore keywords

    Args:
        dct (Dictionary): gensim.corpora Dictionary  a reference Dictionary
        tfidf (list of tfidf):  model[doc]  [(int, number)]
        threshold (float) : high TFIDF socore must be greater than the threshold
        topk(int): num of highest TFIDF socore
    Returns:
        (list): A list of keywords
    """

    tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
    return list(islice(
        [dct[w] for w, score in tfidf if score > threshold], topk
    ))


def create_dataset(data_filepath):
    """1 提取特征列，抛出病情描述为空的数据
       2 保持原有分布
    """
    df = pd.read_csv(data_filepath)
    df.dropna(subset=['CURRENT_MEDICAL_HISTORY', "PHYSICAL_EXAMINATION"], inplace=True)
    pass


def create_dataset_sample(data_filepath,train_filepath,test_filepath,word2vec_filepath):
    """1 提取特征列，抛出病情描述为空的数据
       2 对少数据的病症上采样（回译,换词）
       3 对多数据的病症进行下采样 根据字数分布,108-250
    """
    df = pd.read_csv(data_filepath)
    df.dropna(subset=['CURRENT_MEDICAL_HISTORY', "PHYSICAL_EXAMINATION"], inplace=True)
    df["CURRENT_MEDICAL_HISTORY_wordsize"] = df['CURRENT_MEDICAL_HISTORY'].apply(lambda x: len(str(x)))
    df["PHYSICAL_EXAMINATION_wordsize"] = df['PHYSICAL_EXAMINATION'].apply(lambda x: len(str(x)))

    sample = np.random.choice(df.index, size=int(len(df) * 0.7), replace=False)

    train_df = df.loc[sample]
    test_df = df.drop(sample)

    test_df.to_csv(test_filepath)
    # 急性上呼吸道 的主诉字数分布 去下采样的0.3倍 80-200
    # 支气管炎 的主诉的字数分布 取下采样0.3倍 80-200
    class0 = train_df[(train_df["Pre_Examination_Main_Diag_Name"] == "急性上呼吸道感染") & (
        train_df["CURRENT_MEDICAL_HISTORY_wordsize"].isin(range(80, 201)))]
    class1 = train_df[(train_df["Pre_Examination_Main_Diag_Name"] == "支气管炎") & (
        train_df["CURRENT_MEDICAL_HISTORY_wordsize"].isin(range(80, 201)))]
    class2 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "哮喘"]
    class3 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "咽炎"]
    class4 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "肺炎"]
    class5 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "鼻炎"]
    class6 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "扁桃体炎"]
    class7 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "喉炎"]
    class8 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "鼻窦炎"]
    class9 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "流行性感冒"]
    class10 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "气道异物"]
    class11 = train_df[train_df["Pre_Examination_Main_Diag_Name"] == "疑难杂症"]

    new_df = pd.concat([class0, class1, class3, class4, class5, class11])

    tfidf_model = TfidfModel.load('tfidf.model')
    dct = Dictionary.load('tfidf.dict')
    seg = pkuseg.pkuseg()
    wv = KeyedVectors.load_word2vec_format(word2vec_filepath, binary=False)

    index_add = df.index.tolist()[-1]
    for exp_df in [class2, class8, class9, class6, class7, class10]:
        for _, row in exp_df.iterrows():
            row_list = row.tolist()
            refs = [seg.cut(sample) if sample is not np.nan else seg.cut("") for sample in row_list[23:29]]
            corpus = [dct.doc2bow(doc) for doc in refs]
            new_columns = []

            for token_list, doc in zip(refs, corpus):
                keywords = extract_keywords(dct, tfidf_model[doc])
                num = int(len(token_list) * 0.3)
                new_tokens = token_list.copy()
                while num == int(len(token_list) * 0.3):
                    indexes = np.random.choice(len(token_list), num)
                    for index in indexes:
                        token = token_list[index]
                        if isChinese(token) and token not in keywords and token in wv:
                            new_tokens[index] = wv.most_similar(
                                positive=token, negative=None, topn=1
                            )[0][0]
                    num -= 1

                new_columns.append(''.join(new_tokens))

            row_list[23:29] = new_columns
            index_add += 1
            print(index_add)
            exp_df.loc[index_add] = row_list
        new_df = pd.concat([new_df, exp_df])
    new_df.to_csv(train_filepath)



if __name__ == '__main__':
    # create_dataset_sample()

    data_filepath = ""
    test_filepath = ""
    word2vec_filepath = ""
    train_filepath = ""
    create_dataset()
    create_dataset_sample()
