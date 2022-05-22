import yaml
import numpy as np

from load_files import load_npz_files, load_npz_file
from loss_function import weighted_categorical_cross_entropy
from models.multiConvSleepNet import multiConvSleepNet
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def set_gpu():
    #gpu设置
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)

def init_model(model_path, hyper_param_dict, model_name = "baseCNN"):
    #模型初始化
    print(model_name)
    loss = weighted_categorical_cross_entropy(hyper_param_dict['class_weights'])

    if model_name == "SingleSalientModel":
        eva_model: tensorflow.keras.models.Model = SingleSalientModel(**hyper_param_dict)
    elif model_name == "baseCNN":
        eva_model: tensorflow.keras.models.Model = baseCNN(**hyper_param_dict)
    elif model_name == "TinySleepNet":
        eva_model: tensorflow.keras.models.Model = TinySleepNet(**hyper_param_dict)
    elif model_name == "DeepSleepNet":
        eva_model: tensorflow.keras.models.Model = DeepSleepNet(**hyper_param_dict)
    elif model_name == "BiLSTM":
        eva_model: tensorflow.keras.models.Model = BiLSTM(**hyper_param_dict)
    elif model_name == "LSTM":
        eva_model: tensorflow.keras.models.Model = LSTM(**hyper_param_dict)
    elif model_name == "multiConvSleepNet":
        eva_model: tensorflow.keras.models.Model = multiConvSleepNet(**hyper_param_dict)

    eva_model.compile(optimizer=hyper_param_dict['optimizer'], loss = loss, metrics=['acc'])
    eva_model.load_weights(model_path)
    return eva_model

def predict_lite(input, eva_model, hyper_param_dict):
    #用于lite模型（去掉序列学习部分的）
    input = input[0].reshape(1, 3000, 1, 1)      #[3, 3000, 1, 1]
    #print("output.shape", output.shape)
    pred = eva_model.predict(input, batch_size = hyper_param_dict['batch_size'])
    pred = pred.reshape(hyper_param_dict['sequence_length'], 5)
    pred_label = pred.argmax(axis = 1)
    return pred_label

def predict(input, eva_model, hyper_param_dict):
    #预测
    input = input.reshape(1, hyper_param_dict['sequence_length'], 3000, 1)
    pred = eva_model.predict(input, batch_size = 1)
    pred = pred.reshape(hyper_param_dict['sequence_length'], 5)
    pred_label = pred.argmax(axis = 1)
    return pred_label


