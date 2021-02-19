"""
Usage of this script:
  python train_param_optim_BiLSTM.py <dataset> <method> <classifier> <params_str> <results_dir>
"""
import time
import os
import pandas as pd
from sys import argv
import csv
import keras
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras_self_attention import SeqSelfAttention
from DatasetManager2 import DatasetManager2
import auc_callback
import tensorflow as tf

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# gpu usage amount
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    sess = tf.Session(config=gpu_config)

dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]
params_str = argv[4]
output_dir = argv[5]

params = params_str.split("_")

nnsize = int(params[0])
dropout = float(params[1])
n_layers = int(params[2])
batch_size = int(params[3])
optimizer = params[4]
learning_rate = float(params[5])

activation = "sigmoid"
nb_epoch = 50
train_ratio = 0.8
val_ratio = 0.2

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager2(dataset_name)
data = dataset_manager.read_dataset()
train, _ = dataset_manager.split_data_strict(data, train_ratio)
train, val = dataset_manager.split_val(train, val_ratio)

if "traffic_fines" in dataset_name:
    max_len = 10
elif "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
del data

dt_train = dataset_manager.encode_data_for_lstm(train)
del train
data_dim = dt_train.shape[1] - 3
X, y = dataset_manager.generate_3d_data(dt_train, max_len)
del dt_train

dt_val = dataset_manager.encode_data_for_lstm(val)
del val
X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)
del dt_val

print("Done: %s" % (time.time() - start))

# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()
model = Sequential()
main_input = Input(shape=(max_len, data_dim), name='main_input')

if n_layers == 1:
    lstm1 = LSTM(nnsize, implementation=2, kernel_initializer='glorot_uniform',return_sequences=False, dropout=dropout)
    model.add(keras.layers.Bidirectional(lstm1,input_shape=(max_len, data_dim)))
    model.add(BatchNormalization())

elif n_layers == 2:
    model.add(keras.layers.Bidirectional(LSTM(nnsize, implementation=2, kernel_initializer='glorot_uniform',
                     return_sequences=True, dropout=dropout),input_shape=(max_len, data_dim)))
    model.add(BatchNormalization(axis=1))
    model.add(keras.layers.Bidirectional(LSTM(nnsize, implementation=2, kernel_initializer='glorot_uniform',
                     return_sequences=False, dropout=dropout),merge_mode="concat", weights=None,input_shape=(max_len, data_dim)))
    model.add(BatchNormalization())
elif n_layers == 3:
    model.add(keras.layers.Bidirectional(LSTM(nnsize, implementation=2, kernel_initializer='glorot_uniform',
                     return_sequences=True, dropout=dropout),input_shape=(max_len, data_dim)))
    model.add(BatchNormalization(axis=1))
    model.add(keras.layers.Bidirectional(LSTM(nnsize, implementation=2, kernel_initializer='glorot_uniform',
                     return_sequences=True, dropout=dropout),input_shape=(max_len, data_dim)))
    model.add(BatchNormalization(axis=1))
    model.add(keras.layers.Bidirectional(LSTM(nnsize, implementation=2, kernel_initializer='glorot_uniform',
                     return_sequences=False, dropout=dropout),input_shape=(max_len, data_dim)))
    model.add(BatchNormalization())
model.add(Dense(2, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output'))

if optimizer == "adam":
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
elif optimizer == "rmsprop":
    opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'outcome_output': 'binary_crossentropy'}, optimizer=opt)

auc_cb = auc_callback.AUCHistory(X_val, y_val)
history = model.fit(X, y, validation_data=(X_val, y_val), verbose=2,
                    callbacks=[auc_cb], batch_size=batch_size, epochs=nb_epoch)

print("Done: %s" % (time.time() - start))

# Write loss for each epoch
with open(os.path.join(output_dir, "loss_%s_%s_%s.csv" % (dataset_name, method_name, params_str)), 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
    spamwriter.writerow(["epoch", "train_loss", "val_loss", "val_auc", "params"])
    for epoch in range(len(history.history['loss'])):
        spamwriter.writerow(
            [epoch, history.history['loss'][epoch], history.history['val_loss'][epoch], auc_cb.aucs[epoch],
             params_str])
