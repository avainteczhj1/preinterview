#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import data_preprocess
from textcnn_model_builder import TextCNN
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import pandas as pd
from tensorflow.keras import backend as K
import json

def focal_loss(alpha=0.75, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        ones = K.ones_like(y_true)
        alpha_t = y_true*alpha + (ones-y_true)*(1-alpha)
        p_t = y_true*y_pred + (ones-y_true)*(ones-y_pred) + K.epsilon()
        focal_loss = -alpha_t * K.pow((ones-p_t),gamma) * K.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed


def multi_category_focal_loss2(gamma=2.0, alpha=0.75):
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha,dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        #防止log无限小 导致出现nan情况
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        #alpha为正样本权重
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        #1-pt 为难易区分程度的权重
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


def train(x_train, y_train, vocab_size, feature_size, save_path, embedding_matrix):
    model = TextCNN(vocab_size, feature_size, args.embed_size, args.num_classes,
                    args.num_filters, args.filter_sizes, args.regularizers_lambda, args.dropout_rate,embedding_matrix)
    model.summary()
    # parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    #focal loss
    # model.compile(tf.optimizers.Adam(), loss=multi_category_focal_loss2(alpha=0.75, gamma=2.0),metrics=['accuracy'])
    model.compile(tf.optimizers.Adam(), loss="categorical_crossentropy",metrics=['accuracy'])

    y_train = tf.one_hot(y_train, args.num_classes)
    # tb_callback = keras.callbacks.TensorBoard(os.path.join(args.results_dir, timestamp, 'log/'),
    #                                           histogram_freq=0.1, write_graph=True,
    #                                           write_grads=True, write_images=True,
    #                                           embeddings_freq=0.5, update_freq='batch')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto')
    EarlyStop = EarlyStopping(monitor='val_loss',patience=5, verbose=1, mode='auto')
    history = model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs,
                                 callbacks=[reduce_lr,EarlyStop], validation_split=args.fraction_validation, shuffle=True)
    keras.models.save_model(model, save_path)
    pprint(history.history)


if __name__ == '__main__':
    class_list = []
    label_name = ""
    feature_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--padding_size', default=250, type=int, help='Padding size of sentences.(default=250)')
    parser.add_argument('-e', '--embed_size', default=100, type=int, help='Word embedding size.(default=100)')
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=128, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.5, type=float, help='Dropout rate in softmax layer.(default=0.3)')
    parser.add_argument('-c', '--num_classes', default=12, type=int, help='Number of target classes.(default=12)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float, help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=512, type=int, help='Batch size.(default=64)')
    parser.add_argument('-r', '--epochs', default=50, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.1, type=float, help='The fraction of validation.('
                                                                               'default=0.05)')
    parser.add_argument('--results_dir', default='./results/', type=str, help='The results dir including log, model, vocabulary and some images.(default=./results/)')
    parser.add_argument('--data_file', default='', type=str, help='Dataset file path')
    parser.add_argument('--vocab_file', default='', type=str, help='word vacab file path')
    parser.add_argument('--model_file', default='', type=str, help='word vacab file path')
    args = parser.parse_args()
    print('Parameters:', args, '\n')

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(args.results_dir, timestamp))
    os.mkdir(os.path.join(args.results_dir, timestamp, 'log/'))

    X_train, X_test, Y_train, Y_test, vocab_size = data_preprocess.set_dataset(args.data_file, args.vocab_file, feature_list, class_list, label_name, args.padding_size)
    embedding_matrix = data_preprocess.set_embedding_matrix(args.vocab_file)

    result = pd.value_counts(Y_train)
    print("label分布：",result)
    train(X_train, Y_train, vocab_size, args.padding_size, os.path.join(args.results_dir, timestamp, args.model_file), embedding_matrix)