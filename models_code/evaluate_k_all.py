import os
import argparse
from functools import reduce
import yaml
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

from preprocess import preprocess
#from load_files import train_npzs, valid_npzs, load_npz_files, all_npzs
from load_files import load_npz_files, all_npzs
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

from functools import reduce

def f1_scores_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Using confusion matrix to calculate the f1 score of every class

    :param cm: the confusion matrix

    :return: the f1 score
    """
    def get_tp_rel_sel_from_cm(cm: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        calculate the diagonal, column sum and row sum of the matrix
        """
        tp = np.diagonal(cm)
        rel = np.sum(cm, axis=0)
        sel = np.sum(cm, axis=1)
        return tp, rel, sel

    def precision_scores_from_cm(cm: np.ndarray) -> np.ndarray:
        """
        calculate the precision of matrix
        """
        tp, rel, sel = get_tp_rel_sel_from_cm(cm)
        sel_mask = sel > 0
        precision_score = np.zeros(shape=tp.shape, dtype=np.float32)
        precision_score[sel_mask] = tp[sel_mask] / sel[sel_mask]
        return precision_score

    def recall_scores_from_cm(cm: np.ndarray) -> np.ndarray:
        """
        calculate the recall of matrix
        """
        tp, rel, sel = get_tp_rel_sel_from_cm(cm)
        rel_mask = rel > 0
        recall_score = np.zeros(shape=tp.shape, dtype=np.float32)
        recall_score[rel_mask] = tp[rel_mask] / rel[rel_mask]
        return recall_score

    precisions = precision_scores_from_cm(cm)
    recalls = recall_scores_from_cm(cm)

    # prepare arrays
    dices = np.zeros_like(precisions)

    # Compute dice
    intrs = (2 * precisions * recalls)
    union = (precisions + recalls)
    dice_mask = union > 0
    dices[dice_mask] = intrs[dice_mask] / union[dice_mask]
    return dices




def get_tp_rel_sel_from_cm(cm: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    calculate the diagonal, column sum and row sum of the matrix
    """
    tp = np.diagonal(cm)
    rel = np.sum(cm, axis=0)
    sel = np.sum(cm, axis=1)
    return tp, rel, sel


def precision_scores_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    calculate the precision of matrix
    """
    tp, rel, sel = get_tp_rel_sel_from_cm(cm)
    sel_mask = sel > 0
    precision_score = np.zeros(shape=tp.shape, dtype=np.float32)
    precision_score[sel_mask] = tp[sel_mask] / sel[sel_mask]
    return precision_score

def recall_scores_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    calculate the recall of matrix
    """
    tp, rel, sel = get_tp_rel_sel_from_cm(cm)
    rel_mask = rel > 0
    recall_score = np.zeros(shape=tp.shape, dtype=np.float32)
    recall_score[rel_mask] = tp[rel_mask] / rel[rel_mask]
    return recall_score




def parse_args():
    #parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", '-d', default = "./sleep_data/sleepedf/prepared")
    parser.add_argument("--model_name", '-m', default = 'multiConvSleepNet')

    args = parser.parse_args()

    return args


def summary_models(args: argparse.Namespace, hyper_param_dict: dict):

    model_name = args.model_name
    

    loss = weighted_categorical_cross_entropy(hyper_param_dict['class_weights'])


    #elif model_name == "multiConvSleepNet":
    eva_model: tensorflow.keras.models.Model = multiConvSleepNet(**hyper_param_dict)


    eva_model.compile(optimizer=hyper_param_dict['optimizer'], loss=loss, metrics=['acc'])

    cm_list = []
    acc_list = []
    macro_f1_list = []
    MF1_list = []
    
    pred_list = []
    label_list = []
    #开始循环
    for fold in range(20):
        model_file_name = "k_fold/" + str(fold) + "_" + model_name + "_best_model.h5"
        print("model_file_name:", model_file_name)
        eva_model.load_weights(model_file_name)

        #加载测试数据
        if fold < 13:
            test_npzs = all_npzs[fold*2:fold*2+2]
        elif fold == 13:
            test_npzs = [all_npzs[26]]
        elif fold > 13:
            test_npzs = all_npzs[fold*2-1:fold*2+1]
        print("test_npzs:", test_npzs)
        test_data_list, test_labels_list = load_npz_files(test_npzs)
        test_labels_list = [to_categorical(f) for f in test_labels_list]

        test_data, test_labels = preprocess(test_data_list, test_labels_list, hyper_param_dict, True)

        y_pred = np.array([])
        #if model_name == "multiConvSleepNet":
        y_pred: np.ndarray = eva_model.predict(test_data[0].reshape(-1, hyper_param_dict['sequence_length'], 3000, 1), batch_size = 1) 
            
        y_pred = y_pred.reshape((-1, 5))              #[n, sequence_length, 5]——>[n*sequence_length, 5]
        test_labels = test_labels.reshape((-1, 5))    #[n, sequence_length, 1, 5]——>[n*sequence_length, 5]

        y_pred = y_pred.argmax(axis = 1)
        test_labels = test_labels.argmax(axis = 1)
        
        
        pred_list.extend(y_pred)
        label_list.extend(test_labels)

        
        acc = accuracy_score(test_labels, y_pred)
        macro_f1 = f1_score(test_labels, y_pred, average = 'macro')
        print("acc:", acc)
        print("macro_f1:", macro_f1)
        acc_list.append(acc)
        macro_f1_list.append(macro_f1)

        
        #计算分类别指标
        report = classification_report(test_labels, 
                                       y_pred, target_names = ['W', 'N1', 'N2', 'N3', 'REM'], output_dict = True)
        #print("report:", report)
        #ACC = report['accuracy']
        #print("ACC:", ACC)
        per_class_f1 = [report['W']['f1-score'], report['N1']['f1-score'], 
                           report['N2']['f1-score'], report['N3']['f1-score'], report['REM']['f1-score']]
        MF1 = np.mean(per_class_f1)
        MF1_list.append(MF1)
    
    
        cm = confusion_matrix(test_labels, y_pred)
        cm = np.array(cm)
        #打印不归一化的混淆矩阵
        print(cm)
        cm_list.append(cm)
        eva_model.reset_states()
    
    #退出循环
    sum_cm = reduce(lambda x, y: x + y, cm_list)
    ave_f1 = f1_scores_from_cm(sum_cm)
    ave_acc = np.sum(np.diagonal(sum_cm)) / np.sum(sum_cm)
    print(f"the average accuracy: {ave_acc} and the average f1-score: {ave_f1}")
    MF1 = np.mean(ave_f1)
    print("MF1:", MF1)
    print("macro_f1_list:", macro_f1_list)
    print("MF1_list:", MF1_list)
    print("acc_list:", acc_list)
    mean_macro_f1 = np.mean(macro_f1_list)
    mean_MF1 = np.mean(MF1_list)
    mean_acc = np.mean(acc_list)
    print("mean_macro_f1:", mean_macro_f1)
    print("mean_MF1:", mean_MF1)
    print("mean_acc:", mean_acc)

    
    print("len(pred_list):", len(pred_list))
    print("len(label_list):", len(label_list))
    report = classification_report(label_list, 
                                       pred_list, target_names = ['W', 'N1', 'N2', 'N3', 'REM'], output_dict = True)
    
    per_class_precision = [report['W']['precision'], report['N1']['precision'], 
                           report['N2']['precision'], report['N3']['precision'], report['REM']['precision']]
    per_class_recall = [report['W']['recall'], report['N1']['recall'], 
                           report['N2']['recall'], report['N3']['recall'], report['REM']['recall']]
    per_class_f1 = [report['W']['f1-score'], report['N1']['f1-score'], 
                           report['N2']['f1-score'], report['N3']['f1-score'], report['REM']['f1-score']]
    print("per_class_precision:", per_class_precision)
    print("per_class_recall:", per_class_recall)
    print("per_class_f1:", per_class_f1)
    
    
    #看看是不是一样
    precision2 = precision_scores_from_cm(sum_cm)
    recall2 = recall_scores_from_cm(sum_cm)
    f1_2 = f1_scores_from_cm(sum_cm)
    print("precision2:", precision2)
    print("recall2:", recall2)
    print("f1_2:", f1_2)
    print("sum_cm:", sum_cm)
    
    
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
