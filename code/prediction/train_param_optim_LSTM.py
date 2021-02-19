"""
Usage of this script:
  python train_param_optim_LSTM.py <dataset> <method> <classifier> <params_str> <results_dir>
"""

import time
import os
import csv
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from DatasetManager1 import DatasetManager1
import auc_callback

dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]
params_str = argv[4]
output_dir = argv[5]
params = params_str.split("_")

lstmsize = int(params[0])
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
####读取全部数据获得活动集合和资源集合
dataset_manager = DatasetManager1(dataset_name)
data = dataset_manager.read_dataset()
# split into train and test using temporal split and discard events that overlap the periods
train1, _ = dataset_manager.split_data_strict(data, train_ratio)#return (train, test)
# split into train and test using temporal split
train, val = dataset_manager.split_val(train1, val_ratio)

if "traffic_fines" in dataset_name:
    max_len = 10
elif "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
print(max_len)
#del data
from sklearn import preprocessing
dt_train = dataset_manager.encode_data_for_lstm_part1(train,data)



data_dim = dt_train.shape[1] - 3
X, y = dataset_manager.generate_3d_data_org(dt_train, max_len)
dt_val = dataset_manager.encode_data_for_lstm_part1(val,data)

del train
del dt_train
del val
X_val, y_val = dataset_manager.generate_3d_data_org(dt_val, max_len)


del dt_val

print("Done: %s"%(time.time() - start))


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()

main_input = Input(shape=(max_len, data_dim), name='main_input')

if n_layers == 1:
    l2_3 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(main_input)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 2:
    l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input)
    b1 = BatchNormalization(axis=1)(l1)
    l2_3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b1)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 3:
    l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input)
    b1 = BatchNormalization(axis=1)(l1)
    l2 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(b1)
    b2 = BatchNormalization(axis=1)(l2)
    l3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b2)
    b2_3 = BatchNormalization()(l3)

outcome_output = Dense(2, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output')(b2_3)

model = Model(inputs=[main_input], outputs=[outcome_output])
if optimizer == "adam":
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
elif optimizer == "rmsprop":
    opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'outcome_output':'binary_crossentropy'}, optimizer=opt)

auc_cb = auc_callback.AUCHistory(X_val, y_val)
history = model.fit({'main_input': X}, {'outcome_output':y}, validation_data=(X_val, y_val), verbose=2, callbacks=[auc_cb], batch_size=batch_size, epochs=nb_epoch)

print("Done: %s"%(time.time() - start))


# The code is to write loss for each epoch
with open(os.path.join(output_dir, "part1_loss_%s_%s_%s.csv" % (dataset_name, method_name, params_str)), 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
    spamwriter.writerow(["epoch", "train_loss", "val_loss", "val_auc", "params"])
    for epoch in range(len(history.history['loss'])):
        spamwriter.writerow([epoch, history.history['loss'][epoch], history.history['val_loss'][epoch], auc_cb.aucs[epoch], 
                             params_str])
