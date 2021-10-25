import re
import pandas as pd
from tensorflow.keras import preprocessing
import numpy as np
import json
from sklearn.model_selection import train_test_split
import random

random.seed(1)


def sexcode2sex(row):
    if row == 1:
        return "男"
    elif row == 2:
        return "女"
    else:
        return "其他"


def save_data(x, y, savepath):
    x_tokenizer_list = []
    with open(savepath, "w", encoding="utf8") as f:
        for n, sentence in enumerate(x):
            sentence_new = ' '.join(list(sentence))
            x_tokenizer_list.append(sentence_new)
            f.write(sentence_new + "||||" + str(y[n]) + "\n")
    return x_tokenizer_list


def set_dataset(data_file, vocab_file, feature_list, class_list, label_name, padding_size):
    savepath = ""
    traindata_path = ""
    testdata_path = ""

    df = pd.read_csv(data_file)
    # df = df[df['complaint'].notnull()]
    df.dropna(subset=['CURRENT_MEDICAL_HISTORY', "PHYSICAL_EXAMINATION", "Sex_Code"], inplace=True)

    new_train_x_pd = df[feature_list]
    new_train_y_pd = df[label_name]
    new_train_x_pd["sex"] = new_train_x_pd["Sex_Code"].apply(sexcode2sex)
    sex = new_train_x_pd["sex"].values
    age = new_train_x_pd["age"].values
    complaint = new_train_x_pd["COMPLAINT"].values
    current_medical_history = new_train_x_pd["CURRENT_MEDICAL_HISTORY"].values
    previous_history = new_train_x_pd["PREVIOUS_HISTORY"].values
    allergy_history = new_train_x_pd["ALLERGY_HISTORY"].values
    family_history = new_train_x_pd["FAMILY_HISTORY"].values
    # 增加体格检查信息
    physical_examination = new_train_x_pd["PHYSICAL_EXAMINATION"].values

    new_x = ["性别:" + str(i[0]) + "年龄:" + str(i[1]) + "岁" + "主诉:" + str(i[2]) + "病症:" + str(i[3]) + "病史:" + str(
        i[4]) + "过敏史:" + str(i[5]) + "家族史:" + str(i[6]) + "查体:" + str(i[7]) for i in
             zip(sex, age, complaint, current_medical_history, previous_history, allergy_history, family_history,
                 physical_examination)]
    new_y = new_train_y_pd.values
    y = [class_list.index(i) for i in new_y]
    x_tokenizer_list = save_data(new_x, new_y, savepath)

    text_preprocesser = preprocessing.text.Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                                     lower=True,
                                                     oov_token="<UNK>", split=' ')
    text_preprocesser.fit_on_texts(x_tokenizer_list)
    x = text_preprocesser.texts_to_sequences(x_tokenizer_list)
    word_dict = text_preprocesser.word_index
    json.dump(word_dict, open(vocab_file, 'w', encoding="utf8"), ensure_ascii=False)
    vocab_size = len(word_dict)
    # max_doc_length = max([len(each_text) for each_text in x])
    x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                             padding='post', truncating='post')
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, shuffle=True, random_state=1)

    save_data(X_train, Y_train, traindata_path)
    save_data(X_test, Y_test, testdata_path)

    return X_train, X_test, Y_train, Y_test, vocab_size


def sentence2id(vocab_file, x, padding_size):
    word_dict = json.load(open(vocab_file, 'r', encoding="utf8"))
    vocabulary = word_dict.keys()
    x = [[word_dict[each_word] if each_word in vocabulary else 1 for each_word in each_sentence.split()] for
         each_sentence in x]
    x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                             padding='post', truncating='post')
    return x


def converge_sentence_into_document_from_setted_data(X_train, X_test, vocab_file):
    # 将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
    # 不够的补零，多余的删除，并保存到最终的数据集文件之中
    X_train_seq = []
    X_test_seq = []
    max_sent_in_doc = 5
    max_word_in_sent = 40
    UNKNOWN = 1
    # padding_token = '<unk>'
    # 计算截取区间，通过list存储(ex: [[0, 40], [40, 80], [80, 120], [120, 160], [160, 200]])
    range_list = [0] + [(doc_len + 1) * max_word_in_sent for doc_len in range(max_sent_in_doc)]
    range_list = [[range_list[index], range_list[index + 1]] for index in range(len(range_list) - 1)]
    vocab = json.load(open(vocab_file, 'r', encoding="utf8"))

    for sentece in X_train:
        doc = []
        for range_item in range_list:
            doc.append([word_id for word_id in sentece[range_item[0]:range_item[1]]])
        X_train_seq.append(doc)
    X_train_seq = np.array(X_train_seq)

    for sentece in X_test:
        doc = []
        for range_item in range_list:
            doc.append([word_id for word_id in sentece[range_item[0]:range_item[1]]])
        X_test_seq.append(doc)
    X_test_seq = np.array(X_test_seq)

    return X_train_seq, X_test_seq


def set_embedding_matrix(vocab_file, embedding_path):
    # 加载预训练后的字向量
    # word_dic
    embeddings_index = {}
    with open(vocab_file, "r", encoding="utf8") as f:
        vocab = eval(f.readlines(1)[0])

    with open(embedding_path, "r", encoding="utf8") as f:
        for wordvec in f.readlines():
            wordvec = wordvec.split(" ")
            wordvec[-1] = wordvec[-1].replace("\n", "")
            word = wordvec[0]
            coefs = np.asarray(wordvec[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(vocab) + 1, 100))
    print("embedding_matrix shape", embedding_matrix.shape)
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    with open("data/embedding", "w", encoding="utf8") as f:
        for i in embedding_matrix:
            f.write(str(i) + "\n")
    return embedding_matrix
