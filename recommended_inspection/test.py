import tensorflow as tf
from sklearn import metrics
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
import copy
import matplotlib.pyplot as plt
from itertools import cycle
from tensorflow.keras import backend as K


def multi_category_focal_loss2(gamma=2., alpha=.75):
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # 防止log无限小 导致出现nan情况
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # alpha为正样本权重
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        # 1-pt 为难易区分程度的权重
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    N = (-1) * K.sum(K.round(K.clip(y_true - K.ones_like(y_true), -1, 0)))  # N
    TN = K.sum(K.round(K.clip((y_true - K.ones_like(y_true)) * (y_pred - K.ones_like(y_pred)), 0, 1)))  # TN
    FP = N - TN
    precision = TP / (TP + FP + K.epsilon())  # TT/P
    return precision


def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P - TP  # FN=P-TP
    recall = TP / (TP + FN + K.epsilon())  # TP/(TP+FN)
    return recall


def get_acc(y_true, y_pred):
    y_pred = tf.where(y_pred > 0.5, 1.0, 0.0)
    y_intersection = y_true * y_pred
    y_contain = K.all(y_intersection == y_true, axis=1)
    if_true = tf.where(y_contain)
    acc = tf.size(if_true) / tf.size(y_contain)
    acc = tf.cast(acc, tf.float32)
    return acc


def read_data(target, test_file_path):
    test_list = []
    with open(test_file_path, "r", encoding="utf8") as f:
        for i in f.readlines():
            i = i.strip("\n")
            i = i.split("||||")
            label_i = [int(i) for i in i[1].strip("[").strip("]").split(" ")]
            feature_i = [int(i) for i in i[0].split(" ")]
            target.append(np.array(label_i))
            test_list.append(feature_i)
    return test_list


def topn(test_list, target):
    # top n 准确率计算
    predict_prob = model_h5.predict(np.array(test_list))
    score = copy.deepcopy(predict_prob)
    threshold = 0.5
    predict_prob[predict_prob > threshold] = 1
    predict_prob[predict_prob <= threshold] = 0

    y_intersection = np.array(target) * predict_prob
    y_contain = K.all(y_intersection == np.array(target), axis=1)
    acc = len(y_contain[y_contain == True]) / len(y_contain)
    return score, acc, predict_prob


if __name__ == '__main__':

    model_path = ""
    test_file_path = ""
    file_path = ""
    multilabel = []
    model_h5 = tf.keras.models.load_model(model_path, custom_objects={
        "multi_category_focal_loss2_fixed": multi_category_focal_loss2(gamma=2., alpha=.75),
        "get_acc": get_acc, "getPrecision": getPrecision, "getRecall": getRecall})

    label_list = []
    test_list = read_data(label_list, test_file_path)
    score, acc, predict_prob = topn(test_list, label_list)

    fpr = dict()
    tpr = dict()
    threshold_dict = dict()
    auc_dict = dict()
    with open(file_path, "w", encoding="utf8") as f:
        for i in range(10):
            check = predict_prob[:, i]
            label = np.array(label_list)[:, i]
            res = metrics.confusion_matrix(label, check)
            report = classification_report(label, check)
            f.write("第{}项检查{}的res".format(i, multilabel[i]) + str(res) + "\n")
            f.write("第{}项检查{}的report".format(i, multilabel[i]) + str(report) + "\n")
            fpr[i], tpr[i], threshold_dict[i] = roc_curve(label, score[:, i])
            auc_dict[i] = auc(fpr[i], tpr[i])
            # 画roc 单类
            # plt.figure()
            # plt.plot(fpr[i], tpr[i],
            #          lw=2, label='ROC curve (area = %0.2f)' % auc_dict[i])
            # plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title(f"ROC for {i} class")
            # plt.legend(loc="lower right")
            # plt.savefig("results/2021-04-12-10-26/roc/{}的ROC.png".format(multilabel[i]))

        # 计算多类别的roc
        fpr["micro"], tpr["micro"], _ = roc_curve(np.array(label_list).ravel(), score.ravel())
        auc_dict["micro"] = auc(fpr["micro"], tpr["micro"])
        # auc_dict["micro"] = roc_auc_score(fpr["micro"], tpr["micro"],average="micro")

        # 计算宏平均ROC曲线和AUC

        # 首先汇总所有FPR
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

        # 然后再用这些点对ROC曲线进行插值
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(10):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # 最后求平均并计算AUC
        mean_tpr /= 10

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        auc_dict["macro"] = auc(fpr["macro"], tpr["macro"])

        # 绘制所有ROC曲线
        plt.figure(figsize=(12, 8))
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(auc_dict["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(auc_dict["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'magenta', 'brown'])
        for i, color in zip(range(10), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, auc_dict[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC to multi-class')
        plt.legend(loc="lower right")
        plt.savefig("")

        res = metrics.confusion_matrix(np.array(label_list).flatten(), predict_prob.flatten())
        report = classification_report(np.array(label_list).flatten(), predict_prob.flatten())
        f.write("res\n" + str(res) + "\n")
        f.write("report\n" + str(report) + "\n")
        f.write("fpr\n" + str(fpr) + "\n")
        f.write("tpr\n" + str(tpr) + "\n")
        f.write("auc\n" + str(auc_dict) + "\n")
        f.write("threshold_dict\n" + str(threshold_dict) + "\n")
        f.close()
