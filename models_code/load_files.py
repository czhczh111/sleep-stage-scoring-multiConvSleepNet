import numpy as np


def load_npz_file(npz_file: str) -> (np.ndarray, np.ndarray, int):
    #从npz文件加载数据.
    #return: 一个PSG数据，labels和sampling rate的tuple
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_npz_files(npz_files: list) -> (list, list):
    #从npz文件加载数据.
    #返回data和label的list
    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        
        tmp_data = tmp_data[:, :, :, np.newaxis, np.newaxis]  #[None, W, C, H, N]
        #print("tmp_data.shape:", tmp_data.shape)   #[n, 3000, 3, 1, 1]
        tmp_data = np.concatenate((tmp_data[np.newaxis, :, :, 0, :, :], tmp_data[np.newaxis, :, :, 1, :, :],
                                   tmp_data[np.newaxis, :, :, 2, :, :]), axis=0)  #[None, W, C, H, N]——>[C, None, W, H, N]
        #print("tmp_data.shape:", tmp_data.shape)    #[3, n, 3000, 1, 1]

        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    print(f"load {len(data_list)} files totally.")

    return data_list, labels_list




train_npzs = [ './sleep_data/sleepedf/prepared/SC4001E0.npz',
          './sleep_data/sleepedf/prepared/SC4002E0.npz',
          './sleep_data/sleepedf/prepared/SC4011E0.npz',
          './sleep_data/sleepedf/prepared/SC4012E0.npz',
          './sleep_data/sleepedf/prepared/SC4021E0.npz',
          './sleep_data/sleepedf/prepared/SC4022E0.npz',
          './sleep_data/sleepedf/prepared/SC4032E0.npz',
          './sleep_data/sleepedf/prepared/SC4041E0.npz',
          './sleep_data/sleepedf/prepared/SC4042E0.npz',
          './sleep_data/sleepedf/prepared/SC4052E0.npz',
          './sleep_data/sleepedf/prepared/SC4061E0.npz',
          './sleep_data/sleepedf/prepared/SC4062E0.npz',
          './sleep_data/sleepedf/prepared/SC4071E0.npz',
          './sleep_data/sleepedf/prepared/SC4072E0.npz',
          './sleep_data/sleepedf/prepared/SC4081E0.npz',
          './sleep_data/sleepedf/prepared/SC4082E0.npz',
          './sleep_data/sleepedf/prepared/SC4091E0.npz',
          './sleep_data/sleepedf/prepared/SC4092E0.npz',
          './sleep_data/sleepedf/prepared/SC4101E0.npz',
          './sleep_data/sleepedf/prepared/SC4102E0.npz',
          './sleep_data/sleepedf/prepared/SC4111E0.npz',
          './sleep_data/sleepedf/prepared/SC4112E0.npz',
          './sleep_data/sleepedf/prepared/SC4121E0.npz',
          './sleep_data/sleepedf/prepared/SC4131E0.npz',
          './sleep_data/sleepedf/prepared/SC4141E0.npz',
          './sleep_data/sleepedf/prepared/SC4142E0.npz',
          './sleep_data/sleepedf/prepared/SC4151E0.npz',
          './sleep_data/sleepedf/prepared/SC4152E0.npz',
          './sleep_data/sleepedf/prepared/SC4161E0.npz',
          './sleep_data/sleepedf/prepared/SC4162E0.npz',
          './sleep_data/sleepedf/prepared/SC4171E0.npz',
          './sleep_data/sleepedf/prepared/SC4181E0.npz',
          './sleep_data/sleepedf/prepared/SC4182E0.npz',
          './sleep_data/sleepedf/prepared/SC4191E0.npz',
          './sleep_data/sleepedf/prepared/SC4192E0.npz']

valid_npzs = ['./sleep_data/sleepedf/prepared/SC4031E0.npz',
              './sleep_data/sleepedf/prepared/SC4051E0.npz',
              './sleep_data/sleepedf/prepared/SC4122E0.npz',
              './sleep_data/sleepedf/prepared/SC4172E0.npz'
             ]