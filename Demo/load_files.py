import numpy as np


def load_npz_file(npz_file: str) -> (np.ndarray, np.ndarray, int):
    #从npz文件加载数据.
    #npz_file: a str of npz filename
    #return: 一个PSG数据，labels和sampling rate的tuple
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_npz_files(npz_files: list) -> (list, list):

    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        

        #print("tmp_data.shape:", tmp_data.shape)   #[n, 3000, 3]
        tmp_data = tmp_data[:, :, :, np.newaxis, np.newaxis]
        #print("tmp_data.shape:", tmp_data.shape)   #[n, 3000, 3, 1, 1]
        tmp_data = np.concatenate((tmp_data[np.newaxis, :, :, 0, :, :], tmp_data[np.newaxis, :, :, 1, :, :],
                                   tmp_data[np.newaxis, :, :, 2, :, :]), axis=0)
        #print("tmp_data.shape:", tmp_data.shape)    #[3, n, 3000, 1, 1]


        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    return data_list, labels_list



