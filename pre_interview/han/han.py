#!/usr/bin/python
# -*- coding: UTF-8 -*-
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, TimeDistributed,Masking,Dropout,LayerNormalization

from attention import Attention


class HAN(Model):
    def __init__(self,
                 maxlen_sentence,
                 maxlen_word,
                 max_features,
                 embedding_dims,
                 embedding_matrix,
                 class_num=1,
                 last_activation='sigmoid'):
        super(HAN, self).__init__()
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        # Word part
        input_word = Input(shape=(self.maxlen_word,))
        x_word = Embedding(self.max_features+1, self.embedding_dims, input_length=self.maxlen_word,weights=[embedding_matrix], trainable=True,mask_zero=True)(input_word)

        x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)  # LSTM or GRU
        x_word = LayerNormalization(axis = -1)(x_word)
        x_word = Dropout(0.5)(x_word)
        x_word = Attention(self.maxlen_word)(x_word)
        model_word = Model(input_word, x_word)
        # Sentence part
        self.word_encoder_att = TimeDistributed(model_word)
        self.sentence_encoder = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.sentence_encoder_dropout = Dropout(0.5)
        # mask
        # self.mask = Masking(mask_value=0.0)
        self.sentence_att = Attention(self.maxlen_sentence)

        # Output part
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of HAN must be 3, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen_sentence:
            raise ValueError('The maxlen_sentence of inputs of HAN must be %d, but now is %d' % (self.maxlen_sentence, inputs.get_shape()[1]))
        if inputs.get_shape()[2] != self.maxlen_word:
            raise ValueError('The maxlen_word of inputs of HAN must be %d, but now is %d' % (self.maxlen_word, inputs.get_shape()[2]))
        x_sentence = self.word_encoder_att(inputs)
        x_sentence = self.sentence_encoder(x_sentence)
        x_sentence = self.sentence_encoder_dropout(x_sentence)
        # mask
        # mask = self.mask(x_sentence)
        # x_sentence = self.sentence_att(x_sentence,mask)
        x_sentence = self.sentence_att(x_sentence)

        output = self.classifier(x_sentence)
        return output