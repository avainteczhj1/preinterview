import tensorflow as tf
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from han import HAN
import json
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
    predict_prob = model.predict(test_list)
    prob = np.argsort(predict_prob, axis=-1)[:, ::-1]
    count3 = 0
    count5 = 0
    total_count = len(label_list)
    for m, i in enumerate(label_list):
        if i in prob[m][:3]:
            count3 += 1

        if i in prob[m][:5]:
            count5 += 1

    top3_acc = count3 / total_count
    top5_acc = count5 / total_count
    print("top3_acc:", top3_acc)
    print("top5_acc:", top5_acc)
    return top3_acc, top5_acc


if __name__ == '__main__':
    vocab_file = ''
    embedding_path = ''
    model_weight = ''

    vocab_dict = json.load(open(vocab_file, 'r', encoding="utf8"))
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    vocab_size = len(json.load(open(vocab_file, 'r', encoding="utf8")))
    embedding_matrix = data_preprocess.set_embedding_matrix(vocab_file, embedding_path)
    model = HAN(5, 40, vocab_size, 100, embedding_matrix, 12, 'softmax')
    model.load_weights(model_weight)

    label_list = []
    test_list = []
    test_data_path = ""
    with open(test_data_path, "r", encoding="utf8") as f:
        for i in f.readlines():
            i = i.strip("\n")
            i = i.split("||||")
            label_list.append(int(i[1]))
            test_list.append([int(i) if vocab_dict.get(int(i)) else 0 for i in i[0].split(" ")])

    X_train_seq, X_test_seq = data_preprocess.converge_sentence_into_document_from_setted_data(test_list[1:3],
                                                                                               test_list, vocab_file)
    # X_train_seq,X_test_seq = data_preprocess.converge_sentence_into_document_from_setted_data([], X_test, vocab_file)
    print(len(X_train_seq), len(X_test_seq))
    top3, top5 = top_acc()

    predict = np.argmax(model.predict(X_test_seq), axis=1)
    res = metrics.confusion_matrix(label_list, predict)
    report = classification_report(label_list, predict)
    print(res)
    print(report)
