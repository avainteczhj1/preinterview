#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow import keras

def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters,
            filter_sizes, regularizers_lambda, dropout_rate,embedding_matrix):
    inputs = keras.Input(shape=(feature_size,), name='input_data')
    # embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
    embed = keras.layers.Embedding(vocab_size+1, embed_size, input_length=feature_size, weights=[embedding_matrix], trainable=True)(inputs)
    embed = keras.layers.Reshape((feature_size, embed_size, 1), name='embedding')(embed)

    pool_outputs = []
    for filter_size in list(map(int, filter_sizes.split(','))):
        filter_shape = (filter_size, embed_size)
        conv = keras.layers.Conv2D(num_filters, filter_shape, strides=(1, 1), padding='valid',
                                   data_format='channels_last', activation='relu',
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=keras.initializers.constant(0.1),
                                   # kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                   # bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                   name='convolution_{:d}'.format(filter_size))(embed)
        bn = keras.layers.BatchNormalization(name="bn")(conv)
        max_pool_shape = (feature_size - filter_size + 1, 1)
        pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                      strides=(1, 1), padding='valid',
                                      data_format='channels_last',
                                      name='max_pooling_{:d}'.format(filter_size))(bn)

        pool_outputs.append(pool)
    #合并
    pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
    #展平全连接
    pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
    # pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(pool_outputs)


    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

