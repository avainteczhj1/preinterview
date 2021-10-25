#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from data_utils import read_samples, isChinese, write_samples
import os
from gensim import matutils
from itertools import islice
import numpy as np
import pkuseg

def is_number(num):
    """
    判断字符串是否为数字
    :param num: 目标字符串（判断是否为数字）
    :return: success_flag, api_response
    """
    try:
        float(num)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(num)
        return True
    except (TypeError, ValueError):
        pass

    return False

class EmbedReplace():
    def __init__(self, sample_path, wv_path):
        self.seg = pkuseg.pkuseg()
        self.samples, self.labels = read_samples(sample_path)
        self.refs_tfidf = []
        self.important_list = ['性别', '男', '女', '主诉', '病症', '病史', '过敏史', '家族史', '过敏', '家族', '史']
        for sample in self.samples:
            tokens = []
            for token in self.seg.cut(sample):
                if not is_number(token) and token not in self.important_list:
                    tokens.append(token)
            self.refs_tfidf.append(tokens)
        self.refs = [self.seg.cut(sample) for sample in self.samples]
        self.wv = KeyedVectors.load_word2vec_format(
            wv_path,
            binary=False)

        if os.path.exists('tfidf.model'):
            self.tfidf_model = TfidfModel.load('tfidf.model')
            self.dct = Dictionary.load('tfidf.dict')
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs_tfidf]
        else:
            self.dct = Dictionary(self.refs)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs_tfidf]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('tfidf.dict')
            self.tfidf_model.save('tfidf.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):
        '''
        docs :: iterable of iterable of (int, number)
        '''
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):

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

    def replace(self, token_list, doc):
        """replace token by another token which is similar in wordvector 

        Args:
            token_list (list): reference token list
            doc (list): A reference represented by a word bag model
        Returns:
            (str):  new reference str
        """
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        num = int(len(token_list) * 0.3)
        new_tokens = token_list.copy()
        while num == int(len(token_list) * 0.3):
            indexes = np.random.choice(len(token_list), num)
            for index in indexes:
                token = token_list[index]
                if isChinese(token) and token not in keywords and token in self.wv:
                    new_tokens[index] = self.wv.most_similar(
                        positive=token, negative=None, topn=1
                        )[0][0]
            num -= 1

        return ''.join(new_tokens)

    def generate_samples(self, write_path):
        """generate new samples file
        Args:
            write_path (str):  new samples file path

        """
        replaced = []
        count = 0
        for sample, token_list, doc in zip(self.samples, self.refs, self.corpus):
            count += 1
            ##分批写入，取余符号后的数字表示多少行写一次，不足数量的部分不会写入
            #if count % 10 == 0:
                #print(count)
                #write_samples(replaced, write_path, 'a')
                #replaced = []
            #replaced.append(self.replace(token_list, doc))
            
            #一次性写入，比较耗内存，酌情选择
            replaced.append(self.replace(token_list, doc))
        write_samples(replaced, self.labels, write_path, 'a')


sample_path = 'data/train_oversample.txt'
wv_path = 'data/medical_word_vec_100.bin'
replacer = EmbedReplace(sample_path, wv_path)
replacer.generate_samples('data/embed_replaced.txt')
