# sleep-stage-scoring-multiConvSleepNet

Demo:  
    demo.py: demo，读取存好的EEG，预测标签并通过蓝牙发送给arduino  
    loss_function: 损失函数  
    load_files: 加载npz  
    sleep_stage_predict.py: 部分函数定义  
    models: 模型定义（TinySleepNet, SalientModel, multiConvSleepNet, DeepSleepNet, baseCNN, LSTM, BiLSTM）\\
    hyper_parameters: 各模型对应的超参数定义\\
    saved_deep_models: demo里使用的训练好的模型\\
    test: 提前保存的用于demo的数据（npz文件）\\

Demo_Arduino:
    Demo_Arduino.ino:Arduino代码，包含函数功能包括点亮rgb-led，蓝牙接收数据，渐变调整灯光颜色，实时在oled上显示相关信息等

models_code:
    train.py: 训练
    evaluate.py: 评估
    preprocess.py: 预处理
    loss_function: 损失函数
    load_files: 加载npz文件
    models: 模型定义（TinySleepNet, SalientModel, multiConvSleepNet, DeepSleepNet, baseCNN, LSTM, BiLSTM）
    hyper_parameters: 各模型对应的超参数定义
    prepare_npz: 数据集原始文件加载和预处理
    sleep_data: 处理好的数据集文件
    result: 存储的模型
    preprocess_visualize: 预处理EEG耳机保存的数据 可视化
    KNN.ipynb: 使用KNN分类
    SVM.ipynb: 使用SVM分类

demo_video.mp4: 实物效果展示
