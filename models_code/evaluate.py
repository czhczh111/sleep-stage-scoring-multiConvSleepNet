import os
import argparse
from functools import reduce
import yaml
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

from preprocess import preprocess
from load_files import train_npzs, valid_npzs, load_npz_files
from loss_function import weighted_categorical_cross_entropy

from models.SalientModel import SingleSalientModel, TwoSteamSalientModel
from models.baseCNN import baseCNN
from models.TinySleepNet import TinySleepNet
from models.DeepSleepNet import DeepSleepNet
from models.multiConvSleepNet import multiConvSleepNet
from models.BiLSTM import BiLSTM
from models.LSTM import LSTM

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def parse_args():
    #parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", '-d', default = "./sleep_data/sleepedf/prepared")
    parser.add_argument("--model_name", '-m', default = 'SingleSalientModel')
    parser.add_argument("--output_dir", '-o', default='result/')

    args = parser.parse_args()

    return args


def summary_models(args: argparse.Namespace, hyper_param_dict: dict):

    model_name = args.model_name
    output_dir = args.output_dir
    
    model_file_name = output_dir + model_name + "_best_model.h5"
    
    print("model_file_name:", model_file_name)
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


    eva_model.compile(optimizer=hyper_param_dict['optimizer'], loss=loss, metrics=['acc'])

    eva_model.load_weights(model_file_name)

    #加载测试数据
    test_npzs = valid_npzs
    test_data_list, test_labels_list = load_npz_files(test_npzs)
    test_labels_list = [to_categorical(f) for f in test_labels_list]

    test_data, test_labels = preprocess(test_data_list, test_labels_list, hyper_param_dict, True)

    y_pred = np.array([])
    if model_name == "multiConvSleepNet":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, hyper_param_dict['sequence_length'], 3000, 1), batch_size = 1)
    elif model_name == "DeepSleepNet":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, hyper_param_dict['sequence_length'], 3000, 1), batch_size = 1)
    elif model_name == "SingleSalientModel":
        y_pred: np.ndarray = eva_model.predict(test_data[0], batch_size = 1)
    elif model_name == "TinySleepNet":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, hyper_param_dict['sequence_length'], 3000, 1), batch_size = 1)
    elif model_name == "baseCNN":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, 1, 3000, 1), batch_size = 1)
    elif model_name == "BiLSTM":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, hyper_param_dict['sequence_length'], 3000, 1), batch_size = 1)
    elif model_name == "LSTM":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, hyper_param_dict['sequence_length'], 3000, 1), batch_size = 1)    
            
    y_pred = y_pred.reshape((-1, 5))              #[n, sequence_length, 5]——>[n*sequence_length, 5]
    test_labels = test_labels.reshape((-1, 5))    #[n, sequence_length, 1, 5]——>[n*sequence_length, 5]

    y_pred = y_pred.argmax(axis = 1)
    test_labels = test_labels.argmax(axis = 1)

    acc = accuracy_score(test_labels, y_pred)
    macro_f1 = f1_score(test_labels, y_pred, average = 'macro')
    print("acc:", acc)
    print("macro_f1:", macro_f1)
    cm = confusion_matrix(test_labels, y_pred)

    cm = np.array(cm)
    print("confusion matrix:")
    #打印归一化的混淆矩阵
    print(cm.astype('float32') / np.sum(cm).astype('float32'))
    #打印不归一化的混淆矩阵
    print(cm)

    eva_model.reset_states()
    
    #计算分类别指标
    report = classification_report(test_labels, y_pred, target_names = ['W', 'N1', 'N2', 'N3', 'REM'], output_dict = True)
    print("report:", report)
    ACC = report['accuracy']
    print("ACC:", ACC)
    per_class_precision = [report['W']['precision'], report['N1']['precision'], 
                           report['N2']['precision'], report['N3']['precision'], report['REM']['precision']]
    per_class_recall = [report['W']['recall'], report['N1']['recall'], 
                           report['N2']['recall'], report['N3']['recall'], report['REM']['recall']]
    per_class_f1 = [report['W']['f1-score'], report['N1']['f1-score'], 
                           report['N2']['f1-score'], report['N3']['f1-score'], report['REM']['f1-score']]
    MF1 = np.mean(per_class_f1)
    print("MF1:", MF1)
    print("per_class_precision:", per_class_precision)
    print("per_class_recall:", per_class_recall)
    print("per_class_f1:", per_class_f1)
    

def set_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    
if __name__ == '__main__':
    set_gpu()
    args = parse_args()
    model_name = args.model_name
    param_file = ""
    if model_name == "SingleSalientModel":
        param_file = "SalientParams.yaml"
    elif model_name == "multiConvSleepNet":
        param_file = "multiConvSleepNetParams.yaml"
    elif model_name == "DeepSleepNet":
        param_file = "DeepSleepNetParams.yaml"
    elif model_name == "TinySleepNet":
        param_file = "TinySleepNetParams.yaml"
    elif model_name == "baseCNN":
        param_file = "baseCNNParams.yaml"
    elif model_name == "BiLSTM":
        param_file = "lstmParams.yaml"
    elif model_name == "LSTM":
        param_file = "lstmParams.yaml"
    with open("hyper_parameters/" + param_file, encoding = 'utf-8') as f:
        hyper_params = yaml.full_load(f)

    summary_models(args, hyper_params)
