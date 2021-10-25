# 自然语言分类
import numpy as np
import os

os.environ['TF_KERAS'] = '1'
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tensorflow.keras.layers import Lambda, Dense
from sklearn.model_selection import train_test_split

def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    text_list = []
    text_label = []

    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('||||')
            text = text.replace(" ", "")
            label = [int(i) for i in label.strip("[").strip("]").split(" ")]

            text_list.append(text)
            text_label.append(label)

    X_train, X_test, Y_train, Y_test = train_test_split(text_list, text_label)
    train_data = [(i, Y_train[n]) for n, i in enumerate(X_train)]
    test_data = [(i, Y_test[n]) for n, i in enumerate(X_test)]

    return train_data, test_data


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        threshold = 0.5
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0

        # y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(test_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('/results/best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    set_gelu('tanh')
    num_classes = 10
    maxlen = 128
    batch_size = 32
    epochs = 10
    #需要下载albert  albert_base_google_zh_additional_36k_steps
    config_path = r''
    checkpoint_path = r''
    dict_path = r''

    # 加载数据集
    train_data, test_data = load_data('')
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='albert',
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = Dense(
        units=10,
        activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model(bert.model.input, output)
    model.summary()
    AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5), metrics=['binary_accuracy'])

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)
    evaluator = Evaluator()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    model.load_weights('/results/best_model.weights')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))
