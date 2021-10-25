import joblib
import data_preprocess
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

# logistic_model_adjust = ""
gbdt_model_adjust=''
vocab_file = r"vocab.txt"
padding_size = 200
data_file = r""
feature_list = []
class_list = []
label_name = ""
X_train, X_test, Y_train, Y_test, label2id = data_preprocess.set_dataset_sif(data_file, vocab_file, feature_list, class_list, label_name, padding_size)
# sif embedding
w2vfile = ''  # word vector file, can be downloaded from GloVe website
weightfile = ''  # each line is a word and its frequency
weightpara = 3e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]

# 进行embedding的目标必须是分词完成后的语料（字符串类型，空格隔开每个词）
X_train_sif_embedding = data_preprocess.get_sif_embedding(w2vfile, weightfile, weightpara, X_train)
X_test_sif_embedding = data_preprocess.get_sif_embedding(w2vfile, weightfile, weightpara, X_test)

# lr_model = joblib.load(logistic_model_adjust)
# lr_y_predict_test = lr_model.predict(X_test_sif_embedding)
# lr_y_score = lr_model.decision_function(X_test_sif_embedding)

gbdt_model = joblib.load(gbdt_model_adjust)
gbdt_y_predict_test = gbdt_model.predict(X_test_sif_embedding)
gbdt_y_score = gbdt_model.decision_function(X_test_sif_embedding)

#top n 准确率计算
predict_prob = gbdt_model.predict_proba(X_test_sif_embedding)
prob = np.argsort(predict_prob,axis=-1)[:,::-1]
count3 = 0
count5 = 0
total_count = len(X_test_sif_embedding)
for m,i in enumerate(Y_test):
    if i in prob[m][:3]:
        count3+=1

    if i in prob[m][:5]:
        count5 += 1

top3_acc = count3/total_count
top5_acc = count5/total_count
