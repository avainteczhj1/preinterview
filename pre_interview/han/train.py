#!/usr/bin/python
# -*- coding: UTF-8 -*-
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import argparse
import os
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time
import pandas as pd
from tensorflow.keras import backend as K
import json

from han import HAN
import data_preprocess


def train(x_train, y_train, vocab_size, save_path, embedding_matrix):
    """
    HAN train step
    Args:
        x_train: x data
        y_train: y data
        vocab_size:
        save_path:
        embedding_matrix:

    Returns:

    """
    model = HAN(args.maxlen_sentence, args.maxlen_word, vocab_size, args.embedding_dims, embedding_matrix,
                args.num_classes, 'softmax')
    model.build(X_train_seq.shape)
    model.summary()
    # parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    # model.compile(tf.optimizers.Adam(), loss=focal_loss(alpha=0.75, gamma=2.0),metrics=['accuracy'])
    model.compile(tf.optimizers.Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

    y_train = tf.one_hot(y_train, args.num_classes)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=0, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto')
    EarlyStop = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')
    history = model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs,
                        callbacks=[EarlyStop, reduce_lr], validation_split=args.fraction_validation, shuffle=True)

    model.save_weights(save_path)
    pprint(history.history)


if __name__ == '__main__':
    class_list = []
    label_name = ""
    feature_list = []

    parser = argparse.ArgumentParser(description='This is the HAN train project.')
    parser.add_argument('--num_classes', default=12, type=int, help='num of label classes')
    parser.add_argument('--maxlen_sentence', default=5, type=int, help='max length of sentence')
    parser.add_argument('--maxlen_word', default=40, type=int, help='length of padded words')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--embedding_dims', default=100, type=int, help='num of embedding dim')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.1, type=float,
                        help='The fraction of validation.(default=0.05)')
    parser.add_argument('--results_dir', default='./results/', type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=./results/)')
    parser.add_argument('--data_file', default='', type=str, help='Input data file ')
    parser.add_argument('--embedding_path', default='', type=str, help='Word embedding file path')
    parser.add_argument('--vocab_file', default='', type=str, help='Vacab file path')
    parser.add_argument('--model_name', default='', type=str, help='Save model name')
    parser.add_argument('--padding_size', default=200, type=int, help='sentence padding size')

    args = parser.parse_args()
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(args.results_dir, timestamp))
    os.mkdir(os.path.join(args.results_dir, timestamp, 'log/'))

    vocab = json.load(open(args.vocab_file, 'r', encoding="utf8"))
    X_train, X_test, Y_train, Y_test, vocab_size = data_preprocess.set_dataset(args.data_file, args.vocab_file,feature_list,
                                                                               class_list, label_name, args.padding_size)
    embedding_matrix = data_preprocess.set_embedding_matrix(args.vocab_file, args.embedding_path)

    X_train_seq, X_test_seq = data_preprocess.converge_sentence_into_document_from_setted_data(X_train, X_test,
                                                                                               args.vocab_file)

    X_train_seq = tf.convert_to_tensor(X_train_seq, tf.float32, name='input_x_train')
    X_test_seq = tf.convert_to_tensor(X_test_seq, tf.float32, name='input_x_test')

    result = pd.value_counts(Y_train)
    print("label分布：", result)
    train(X_train_seq, Y_train, vocab_size, os.path.join(args.results_dir, timestamp, args.model_name),
          embedding_matrix)
