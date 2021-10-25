#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import data_preprocess


def focal_loss(alpha=0.75, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        ones = K.ones_like(y_true)
        alpha_t = y_true * alpha + (ones - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (ones - y_true) * (ones - y_pred) + K.epsilon()
        focal_loss = -alpha_t * K.pow((ones - p_t), gamma) * K.log(p_t)
        return tf.reduce_mean(focal_loss)

    return focal_loss_fixed


# top n 准确率计算
def top_acc():
    predict_prob = model_h5.predict(test_list)
    prob = np.argsort(predict_prob, axis=-1)[:, ::-1]
    count3 = 0
    count5 = 0
    total_count = len(label_list)
    for m, n in enumerate(label_list):
        if n in prob[m][:3]:
            count3 += 1

        if n in prob[m][:5]:
            count5 += 1

    top3_acc = count3 / total_count
    top5_acc = count5 / total_count
    print("top3_acc:", top3_acc)
    print("top5_acc:", top5_acc)
    return top3_acc, top5_acc


if __name__ == '__main__':
    model_path = ""
    tf_model_path = ""
    test_data_path = ""
    vocab_file = r"vocab.txt"
    # model_h5=tf.keras.models.load_model(model_path,custom_objects={"multi_category_focal_loss2_fixed":focal_loss(
    # alpha=0.75, gamma=2.0)})
    model_h5 = tf.keras.models.load_model(model_path)
    # tf.keras.models.save_model(model_h5,tf_model_path)

    label_list = []
    test_list = []
    with open(test_data_path, "r", encoding="utf8") as f:
        for i in f.readlines():
            i = i.strip("\n")
            i = i.split("||||")
            label_list.append(int(i[1]))
            test_list.append([int(i) for i in i[0].split(" ")])

    top3, top5 = top_acc()

    predict = np.argmax(model_h5.predict(test_list), axis=1)
    res = metrics.confusion_matrix(label_list, predict)
    report = classification_report(label_list, predict)
    print(res)
    print(report)
