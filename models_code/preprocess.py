import numpy as np
from concurrent.futures import ThreadPoolExecutor


def normalization(data: np.ndarray) -> np.ndarray:
    #data: PSG数据
    #返回标准化后的PSG数据
    for i in range(data.shape[0]):
        data[i] -= data[i].mean(axis=0)
        data[i] /= data[i].std(axis=0)
    return data


def preprocess(data: list, labels: list, param: dict, not_enhance: bool = False) -> (np.ndarray, np.ndarray):
    
    #将raw PSG处理成可以输入模型的序列
    #data: PSG 数据的list
    #labels: the list of sleep stage labels
    #param: 超参数dict
    #not_enhance: 是否使用数据增强
    #返回数据和标签的序列

    def data_big_group(d: np.ndarray) -> np.ndarray:
        #数据划分
        return_data = np.array([])
        beg = 0
        while (beg + param['sequence_length']) <= d.shape[1]:       #sequence_length = 25    #最后一个截断
            y = d[:, beg: beg + param['sequence_length'], ...]     #d:(3, 841, 3000, 1, 1), y:(3, 40, 3000, 1, 1)
            y = y[:, np.newaxis, ...]                     #y:(3, 1, 40, 3000, 1, 1)
            return_data = y if beg == 0 else np.append(return_data, y, axis = 1) #axis这个维度增加
            beg += param['sequence_length']
        #print("return_data.shape:", return_data.shape)           #(3, 21, 40, 3000, 1, 1)
        shape = return_data.shape
        #print("shape:", shape)
        return_data = return_data.reshape([shape[0], shape[1], -1, shape[4], shape[5]])
        #print("return_data.shape:", return_data.shape)
        return return_data

    def label_big_group(l: np.ndarray) -> np.ndarray:
        #标签划分
        return_labels = np.array([])
        beg = 0
        while (beg + param['sequence_length']) <= len(l):
            y = l[beg: beg + param['sequence_length']]         #l:(841, 5)   y:(40, 5)
            y = y[np.newaxis, ...]                      #y:(1, 40, 5)
            return_labels = y if beg == 0 else np.append(return_labels, y, axis=0)
            beg += param['sequence_length']
        #print("return_labels.shape:", return_labels.shape)       #(21, 40, 5)
        return return_labels




    preprocessed_data = []
    preprocessed_label = []

    for item in data:
        preprocessed_item = data_big_group(item)
        preprocessed_data.append(preprocessed_item)

    for item in labels:
        preprocessed_item = label_big_group(item)
        preprocessed_label.append(preprocessed_item)
    
    final_data = np.concatenate(preprocessed_data, axis = 1)
    print("final_data.shape", final_data.shape)                  #(3, N, seq*3000, 1, 1)
    final_label = np.concatenate(preprocessed_label, axis = 0)
    print("final_label.shape", final_label.shape)                #(N, seq, 5)
    final_label = final_label[:, :, np.newaxis, :]
    
    return final_data, final_label


