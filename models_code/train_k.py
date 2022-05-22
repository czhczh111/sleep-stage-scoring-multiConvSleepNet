import os

import argparse
import yaml
import numpy as np
import tensorflow.keras.models
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import math
import itertools

from preprocess import preprocess
from load_files import load_npz_files, all_npzs

from models.SalientModel import SingleSalientModel, TwoSteamSalientModel
from models.baseCNN import baseCNN
from models.TinySleepNet import TinySleepNet
from models.DeepSleepNet import DeepSleepNet
from models.multiConvSleepNet import multiConvSleepNet
from models.BiLSTM import BiLSTM
from models.LSTM import LSTM

from loss_function import weighted_categorical_cross_entropy

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def set_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)


def get_parser() -> argparse.Namespace:
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", '-m', default = 'SingleSalientModel')
    parser.add_argument("--data_dir", '-d', default = "./sleep_data/sleepedf/prepared")
    parser.add_argument("--output_dir", '-o', default = './result')
    parser.add_argument("--fold", default='0')
    
    args = parser.parse_args()

    res_path = args.output_dir
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    return args


def print_params(params: dict):
    #打印超参数
    print("=" * 20, "[Hyperparameters]", "=" * 20)
    for (key, val) in params.items():
        if isinstance(val, dict):
            print(f"{key}:")
            for (k, v) in val.items():
                print(f"\t{k}: {v}")
        else:
            print(f"{key}: {val}")
    print("=" * 60)


def train(args: argparse.Namespace, hyper_param_dict: dict) -> dict:

    res_path = args.output_dir

    model_name = args.model_name
    
    fold = eval(args.fold)
    
    #GPU训练
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#     #CPU训练
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    epoch_length = hyper_param_dict['sleep_epoch_length']               #3000
    seq_length = hyper_param_dict['sequence_length']                    #20/25/15
    
    
    #损失函数
    weighted_loss = weighted_categorical_cross_entropy(np.asarray(hyper_param_dict['class_weights']))
    print(f"loss weights: {hyper_param_dict['class_weights']}")


    if model_name == "SingleSalientModel":
        model: tensorflow.keras.models.Model = SingleSalientModel(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc',precision, recall])    #adam初始学习率0.001
    elif model_name == "baseCNN":
        model: tensorflow.keras.models.Model = baseCNN(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc'])

    elif model_name == "TinySleepNet":
        model: tensorflow.keras.models.Model = TinySleepNet(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc',precision, recall])

    elif model_name == "DeepSleepNet":
        model: tensorflow.keras.models.Model = DeepSleepNet(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc',precision, recall])
      

    elif model_name == "BiLSTM":
        model: tensorflow.keras.models.Model = BiLSTM(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc', precision, recall])
    elif model_name == "LSTM":
        model: tensorflow.keras.models.Model = LSTM(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc', precision, recall])

    elif model_name == "multiConvSleepNet":
        model: tensorflow.keras.models.Model = multiConvSleepNet(**hyper_param_dict)
        model.compile(optimizer = hyper_param_dict["optimizer"], loss = weighted_loss, metrics = ['acc', precision, recall])

    #model.summary()    #打印网络结构&参数
        
    #训练
#     npzs_list = np.asarray(all_npzs)
#     npzs_list = np.array_split(npzs_list, 20)
#     print("npzs_list:", npzs_list)

    if fold < 13:
        valid_npzs = all_npzs[fold*2:fold*2+2]
    elif fold == 13:
        valid_npzs = [all_npzs[26]]
    elif fold > 13:
        valid_npzs = all_npzs[fold*2-1:fold*2+1]
    train_npzs = [i for i in all_npzs if i not in valid_npzs]
    
#     #试一下原来的
#     raw_valid_index = [6,10,25,34]
#     valid_npzs = [all_npzs[i] for i in raw_valid_index]
#     train_npzs = [i for i in all_npzs if i not in valid_npzs]
    
    print("train_npzs:", train_npzs)
    print("valid_npzs:", valid_npzs)
    print(type(train_npzs[0]))
    print("len(train_npzs):", len(train_npzs))
    print("len(valid_npzs):", len(valid_npzs))
    
    
    train_data_list, train_labels_list = load_npz_files(train_npzs)
    val_data_list, val_labels_list = load_npz_files(valid_npzs)

    print("len(train_data_list)", len(train_data_list))  #35
    for i in range(len(train_data_list)):
        print("train_data_list[" + str(i) + "].shape", train_data_list[i].shape)    #(3, 841, 3000, 1, 1)
        
    #标签转独热标签
    train_labels_list = [to_categorical(f) for f in train_labels_list]  #[N,] ——>[N, 5]   int32———>float32
    val_labels_list = [to_categorical(f) for f in val_labels_list]
    print("train_labels_list[0].dtype:",train_labels_list[0].dtype)   #float32
    for i in range(len(train_labels_list)):
        print("train_labels_list[i].shape:", train_labels_list[i].shape)  #[1246, 5]
    all_label = np.concatenate(train_labels_list, axis = 0)
    print("all_label.shape:", all_label.shape)


    epochnumber = 0
    for item in train_labels_list:
        epochnumber += item.shape[0]
    for item in val_labels_list:
        epochnumber += item.shape[0]
    print("Total epoch number: ", epochnumber)

    #预处理
    train_data, train_labels = preprocess(train_data_list, train_labels_list, hyper_param_dict, True)
    val_data, val_labels = preprocess(val_data_list, val_labels_list, hyper_param_dict, True)
    
    print("train_data.shape:", train_data.shape)           #(3, 1684, 60000, 1, 1)
    print("train_labels.shape:", train_labels.shape)       #(1684, 20, 1, 5)
 

    #随机打乱数据
    index = [i for i in range(train_data.shape[1])]
    np.random.shuffle(index)
    for j in range(len(train_data)):
        train_data[j] = train_data[j][index]
    train_labels = train_labels[index]

    print(f"train on {train_data.shape[1]} samples, each has {train_data.shape[2] / epoch_length} sleep epochs")
    print(f"validate on {val_data.shape[1]} samples, each has {val_data.shape[2] / epoch_length} sleep epochs")

    #学习率衰减
    def step_decay(epoch):
        init_lr = 0.001
        learning_rate = init_lr*math.pow(0.4, int((epoch)/40))
        return learning_rate
    
    fold = str(fold)
    callback_list = [callbacks.EarlyStopping(monitor = 'acc', patience = hyper_param_dict['patience']),
                     callbacks.ModelCheckpoint(filepath = os.path.join(res_path, f"{fold}_{model_name}_best_model.h5"),
                                               monitor = 'val_acc', save_best_only = True,
                                               save_weights_only = True, period = 1),
                     callbacks.LearningRateScheduler(step_decay, verbose = 1)
                     ]


    if model_name == "SingleSalientModel":  # only EEG: [None, W * gsl, H, N]
        history = model.fit(train_data[0], train_labels, epochs = hyper_param_dict['epochs'],
                            batch_size = hyper_param_dict['batch_size'], callbacks = callback_list,
                            validation_data = (val_data[0], val_labels), verbose = 1)    #verbose=2
        
    elif model_name == "baseCNN":
        history = model.fit(train_data[0].reshape(-1, 1, epoch_length, 1), train_labels.reshape(-1, 5),
                            epochs = hyper_param_dict['epochs'],
                            batch_size = hyper_param_dict['batch_size'], callbacks = callback_list,
                            validation_data = (val_data[0].reshape(-1, 1, epoch_length, 1), val_labels.reshape(-1, 5)), verbose = 1)


    elif model_name == "TinySleepNet":
        history = model.fit(train_data[0].reshape(-1, seq_length, epoch_length, 1), train_labels.reshape(-1, seq_length, 5),
                            epochs = hyper_param_dict['epochs'],
                            batch_size = hyper_param_dict['batch_size'], callbacks = callback_list,
                            validation_data = (val_data[0].reshape(-1, seq_length, epoch_length, 1),
                            val_labels.reshape(-1, seq_length, 5)), verbose = 1)
    elif model_name == "DeepSleepNet":
        history = model.fit(train_data[0].reshape(-1, seq_length, epoch_length, 1), train_labels.reshape(-1, seq_length, 5),
                            epochs = hyper_param_dict['epochs'],
                            batch_size = hyper_param_dict['batch_size'], callbacks = callback_list,
                            validation_data = (val_data[0].reshape(-1, seq_length, epoch_length, 1, 1),
                            val_labels.reshape(-1, seq_length, 5)), verbose = 1)


    elif model_name == "BiLSTM" or model_name == "LSTM":  # only use EEG, the shape is [None, W * gsl, H, N]
        history = model.fit(train_data[0].reshape(-1, seq_length, epoch_length), train_labels.reshape(-1, seq_length, 5),
                            epochs = hyper_param_dict['epochs'],
                            batch_size = hyper_param_dict['batch_size'], callbacks = callback_list,
                            validation_data = (val_data[0].reshape(-1, seq_length, epoch_length),
                            val_labels.reshape(-1, seq_length, 5)), verbose = 1)


    elif model_name == "multiConvSleepNet":  # only use EEG, the shape is [None, W * gsl, H, N]
        history = model.fit(train_data[0].reshape(-1, seq_length, epoch_length, 1), train_labels.reshape(-1, seq_length, 5),
                            epochs = hyper_param_dict['epochs'],
                            batch_size = hyper_param_dict['batch_size'], callbacks = callback_list,
                            validation_data = (val_data[0].reshape(-1, seq_length, epoch_length, 1),
                            val_labels.reshape(-1, seq_length, 5)), verbose = 1)


    K.clear_session()    #销毁当前的 TF 图并创建一个新图，避免旧模型/网络层混乱。
    model.reset_states()


if __name__ == "__main__":
    set_gpu()

    args = get_parser()
    model_name = args.model_name
    param_file = ""

    if model_name == "SingleSalientModel":
        param_file = "SalientParams.yaml"
    elif model_name == "baseCNN":
        param_file = "baseCNNParams.yaml"
    elif model_name == "TinySleepNet":
        param_file = "TinySleepNetParams.yaml"
    elif model_name == "DeepSleepNet":
        param_file = "DeepSleepNetParams.yaml"
    elif model_name == "BiLSTM":
        param_file = "lstmParams.yaml"
    elif model_name == "LSTM":
        param_file = "lstmParams.yaml"
    elif model_name == "multiConvSleepNet":
        param_file = "multiConvSleepNetParams.yaml"

    with open("hyper_parameters/" + param_file, encoding = 'utf-8') as f:
        hyper_params = yaml.full_load(f)
    print_params(hyper_params)

    train_history = train(args, hyper_params)

