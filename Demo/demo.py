# -*- coding: utf-8 -*-
import serial
import time
from sleep_stage_predict import set_gpu, init_model, predict_lite, predict
from load_files import load_npz_files
import yaml
import numpy as np

#serialPort = "COM5"   #串口
serialPort = "COM9"    #蓝牙是COM9
baudRate = 9600        #波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)
print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))

#模型初始化
set_gpu()
test_data_list, _ = load_npz_files(["test/1.npz"])
test_data = test_data_list[0]
# print("test_data_list[0].shape", test_data_list[0].shape)
with open("hyper_parameters/multiConvSleepNetParams.yaml", encoding='utf-8') as f:
    hyper_params = yaml.full_load(f)

model_path = "saved_deep_models/multiConvSleepNet_best_model.h5"
eva_model = init_model(model_path, hyper_params, model_name = "multiConvSleepNet")

#demo1 = b"0" #将数字转换为ASCII码发送

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

c = 0
start_idx = 39  #从哪个idx开始预测
pred_label = 0

test_data_list, _ = load_npz_files(["test/" + str(i + 1) + ".npz" for i in range(start_idx, start_idx + 20)])
test_data_list = [item[0] for item in test_data_list]
pred_list = []
test_data = np.concatenate(test_data_list)
pred_label = predict(test_data, eva_model, hyper_params)
pred_list.extend(pred_label)

idx = 0
while idx <= 100:
    new_test_data, _ = load_npz_files(["test/" + str(start_idx + 20 + idx) + ".npz"])
    new_test_data = new_test_data[0][0]
    test_data_list.append(new_test_data)
    test_data_list.pop(0)
    test_data = np.concatenate(test_data_list)
    pred_labels = predict(test_data, eva_model, hyper_params)
    pred_label = pred_labels[-1]

    start_idx+=1

    #print("pred_label: ", pred_label)
    ascii_pred_label = ord(str(pred_label))
    if ascii_pred_label == 48:
        print("send pred value:", pred_label)
        print("current sleep stage:", class_dict[pred_label])
        ser.write(b"0") #ser.write在于向串口中写入数据
    if ascii_pred_label == 49:
        print("send pred value:", pred_label)
        print("current sleep stage:", class_dict[pred_label])
        ser.write(b"1")
    if ascii_pred_label == 50:
        print("send pred value:", pred_label)
        print("current sleep stage:", class_dict[pred_label])
        ser.write(b"2")
    if ascii_pred_label == 51:
        print("send pred value:", pred_label)
        print("current sleep stage:", class_dict[pred_label])
        ser.write(b"3")
    if ascii_pred_label == 52:
        print("send pred value:", pred_label)
        print("current sleep stage:", class_dict[pred_label])
        ser.write(b"4")

    time.sleep(5)