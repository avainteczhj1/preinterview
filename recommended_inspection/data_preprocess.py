import re
import pandas as pd
from tensorflow.keras import preprocessing
import numpy as np
import json
from sklearn.model_selection import train_test_split
import random


def sexcode2sex(df):
    if df == 1:
        return "男"
    elif df == 2:
        return "女"
    else:
        return "其他"


def set_embedding_matrix(vocab_file):
    # 加载预训练后的字向量
    # word_dic
    embeddings_index = {}
    with open(vocab_file, "r", encoding="utf8") as f:
        vocab = eval(f.readlines(1)[0])

    with open(r"data\medical_token_vec_100.txt", "r", encoding="utf8") as f:
        for wordvec in f.readlines():
            wordvec = wordvec.split(" ")
            wordvec[-1] = wordvec[-1].replace("\n", "")
            word = wordvec[0]
            coefs = np.asarray(wordvec[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(vocab) + 1, 100))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    with open("data/embedding", "w", encoding="utf8") as f:
        for i in embedding_matrix:
            f.write(str(i) + "\n")
    return embedding_matrix


def save_train_test_data(file_path, x, y):
    with open(file_path, "w",encoding="utf8") as f:
        for n, sentence in enumerate(x):
            sentence = [str(x) for x in sentence]
            sentence_new = " ".join(sentence)
            f.write(sentence_new + "||||" + str(y[n]) + "\n")
    pass

def set_dataset(data_file, vocab_file, padding_size):
    df = pd.read_csv(data_file)
    df = df[df['COMPLAINT'].notnull()]
    label = df.loc[:, "血常规":"气道重建"]
    label_num = label.sum()
    sum = label_num[label_num > 500]
    label_new = df.loc[:, sum.index.values.tolist()]
    label_num2 = label_new.sum(axis=1)
    # 有标签的代表有做检验检查
    index = label_num2[label_num2 > 0].index.values
    label_new = df.loc[index, sum.index.values.tolist()]
    df = df.loc[index, :]
    df['sex'] = df['Sex_Code'].apply(sexcode2sex)
    complaint = df["COMPLAINT"].values
    current_medical_history = df["CURRENT_MEDICAL_HISTORY"].values
    previous_history = df["PREVIOUS_HISTORY"].values
    diag_1st_label_illness_name = df["Pre_Examination_Main_Diag_Name"].values
    allergy_history = df["ALLERGY_HISTORY"].values
    family_history = df["FAMILY_HISTORY"].values
    sex = df["sex"].values
    age = df["age"].values
    new_x = ["患病名称" + str(i[0]) + "性别:" + str(i[1]) + "年龄:" + str(i[2]) + "岁，" + "主诉:" + str(i[3]) + " 病症:" + str(
        i[4]) + " 病史:" + str(
        i[5]) + "过敏史:" + str(i[6]) + "家族史:" + str(i[7]) for i in
             zip(diag_1st_label_illness_name, sex, age, complaint, current_medical_history, previous_history,
                 allergy_history, family_history)]

    new_y = label_new.values
    x_tokenizer_list = []
    y = np.array(new_y)

    with open(r"data/dataset", "w", encoding="utf8") as f:
        for n, sentence in enumerate(new_x):
            sentence_new = ' '.join(list(sentence))
            x_tokenizer_list.append(sentence_new)
            f.write(sentence_new + "||||" + str(new_y[n]) + "\n")

    text_preprocesser = preprocessing.text.Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                                     lower=True,
                                                     oov_token="<UNK>", split=' ')
    text_preprocesser.fit_on_texts(x_tokenizer_list)
    x = text_preprocesser.texts_to_sequences(x_tokenizer_list)
    word_dict = text_preprocesser.word_index
    json.dump(word_dict, open(vocab_file, 'w', encoding="utf8"), ensure_ascii=False)
    vocab_size = len(word_dict)
    x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                             padding='post', truncating='post')
    random.seed(1)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, shuffle=False)

    train_datafile_path = r"data\train_data.txt"
    test_datafile_path = r"data\test_data.txt"
    save_train_test_data(train_datafile_path,X_train,Y_train)
    save_train_test_data(test_datafile_path,X_test,Y_test)
    return X_train, X_test, Y_train, Y_test, vocab_size
