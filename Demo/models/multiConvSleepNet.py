from tensorflow.keras import layers, models
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import tensorflow as tf

class multiConvSleepNet(models.Model):
    def __init__(self, padding: str ='same', build: bool = True, **kwargs):
        super(multiConvSleepNet, self).__init__()

        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_length']          #3000
        self.sequence_length = kwargs['sequence_length']                #20

        if build:
            super().__init__(*self.init_model())

    def init_model(self, input: KerasTensor = None) -> (list, list):
        
        cnn_outputs = []
        if input is None:
            input = layers.Input(shape=(20, self.sleep_epoch_length, 1))  ##[20, 3000, 1, 1]
        print("input.shape",input.shape)                         #(None, 20, 3000, 1, 1)——>(None, 20, 3000, 1)
        
        weight_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="normal")
        bias_initializer = tf.zeros_initializer()
        
        ####子卷积网络1
        #conv1
        output = layers.Conv1D(filters = 64, kernel_size = 50,
                               strides = 6, padding = "SAME",
                               use_bias = False, kernel_initializer = weight_initializer,
                               bias_initializer = bias_initializer, name = "small_cnn_conv_1",
                               input_shape = (3000,1))(input)
        print("1output.shape", output.shape)                                #(None, 20, 500, 1, 64)——>(None, 20, 500, 64)
        output = layers.BatchNormalization(name = "small_cnn_batchnorm_1")(output)       #全是默认参数
        print("2output.shape", output.shape)                                 #(None, 20, 500, 1, 64)——>(None, 20, 500, 64)
        output = layers.ReLU(name = "small_cnn_relu_1")(output)                    #(None, 20, 500, 1, 64)——>(None, 20, 500, 64)
        print("3output.shape", output.shape)                                 #(None, 20, 500, 1, 64)——>(None, 20, 500, 64)
        
        output = layers.MaxPool2D(pool_size = (8, 1), strides = (8, 1),           #MaxPooling2D和MaxPool2D都可以
                                  padding = "SAME", name = "small_cnn_maxpool_1",
                                  data_format = 'channels_first')(output)

        print("4output.shape", output.shape)                                 #(None, 20, 63, 1, 64)——>(None, 20, 63, 64)
        output = layers.Dropout(rate = 0.5, name = "small_cnn_dropout")(output)
        print("4output.shape", output.shape)
        
        #conv2
        output = layers.Conv1D(filters = 128, kernel_size = 8,
                               strides = 1, padding = "SAME",
                               use_bias = False, kernel_initializer = weight_initializer,
                               bias_initializer = bias_initializer, name = "small_cnn_conv_2")(output)
        print("5output.shape", output.shape)                                 #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        output = layers.BatchNormalization(name = "small_cnn_batchnorm_2")(output)       #全是默认参数
        print("6output.shape", output.shape)                                 #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        
        #residual
        residual = layers.ReLU(name = "small_cnn_relu_2")(output)                    #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        print("7residual.shape", residual.shape)

        #conv3
        output = layers.SeparableConv2D(filters = 128, kernel_size = (1, 8),
                                        data_format='channels_last',
                                        strides = (1, 1), padding = "SAME",
                                        use_bias = False, kernel_initializer = weight_initializer,
                                        bias_initializer = bias_initializer, name = "small_cnn_conv_3")(residual)
        print("8output.shape", output.shape)                                #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        output = layers.BatchNormalization(name = "small_cnn_batchnorm_3")(output)       #全是默认参数
        print("9output.shape", output.shape)                                #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        output = layers.ReLU(name = "small_cnn_relu_3")(output)                   #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        print("10output.shape", output.shape)
        
        
        #conv4
        output = layers.SeparableConv2D(filters = 128, kernel_size = (1,8),
                                        data_format='channels_last',
                                        strides = 1, padding = "SAME",
                                        use_bias = False, kernel_initializer = weight_initializer,
                                        bias_initializer = bias_initializer, name = "small_cnn_conv_4")(output)    

        print("11output.shape", output.shape)                                #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        output = layers.BatchNormalization(name = "small_cnn_batchnorm_4")(output)       #全是默认参数
        print("12output.shape", output.shape)                                #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        
        #residual
        output = layers.Add(name = "residual_add1")([output, residual])
        print("shape after add residual:", output.shape)
        output = layers.ReLU(name = "small_cnn_relu_4")(output)                   #(None, 20, 63, 1, 128)——>(None, 20, 63, 128)
        print("13output.shape", output.shape)
        
        
        output = layers.MaxPool2D(pool_size = (4, 1), strides = (4, 1),          #MaxPooling2D和MaxPool2D都可以
                          padding = "SAME", name = "small_cnn_maxpool_2",
                          data_format = 'channels_first')(output)
        print("14output.shape", output.shape)                                #(None, 20, 16, 1, 128)——>(None, 20, 16, 128)
        
        
        output = layers.Reshape((output.shape[1], -1))(output)
        print("15output.shape", output.shape)                                #(None, 20, 2048)
        
        cnn_outputs.append(output)
        




        
        #子卷积网络3    
        #conv1
        output2 = layers.Conv1D(filters = 64, kernel_size = 400,
                                strides = 50, padding = "SAME",
                                use_bias = False, kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer, name = "large_cnn_conv_1",
                                input_shape = (3000,1))(input)
        print("1output2.shape", output2.shape)                                #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        output2 = layers.BatchNormalization(name = "large_cnn_batchnorm_1")(output2)       #全是默认参数
        print("2output2.shape", output2.shape)                                 #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        output2 = layers.ReLU(name = "large_cnn_relu_1")(output2)                    #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        print("3output2.shape", output2.shape)                                 #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        
        output2 = layers.MaxPool2D(pool_size = (4, 1), strides = (4, 1),           #MaxPooling2D和MaxPool2D都可以
                                   padding = "SAME", name = "large_cnn_maxpool_1",
                                   data_format = 'channels_first')(output2)

        print("4output2.shape", output2.shape)                                 #(None, 20, 15, 1, 64)——>(None, 20, 15, 64)
        output2 = layers.Dropout(rate = 0.5, name = "large_cnn_dropout")(output2)
        
        #conv2
        output2 = layers.Conv1D(filters = 128, kernel_size = 8,
                                strides = 1, padding = "SAME",
                                use_bias = False, kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer, name = "large_cnn_conv_2")(output2)
        print("5output2.shape", output2.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        output2 = layers.BatchNormalization(name = "large_cnn_batchnorm_2")(output2)        #全是默认参数
        print("6output2.shape", output2.shape)
        
        #residual
        residual2 = layers.ReLU(name = "large_cnn_relu_2")(output2)                    #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        print("7residual2.shape", residual2.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        
        
        #conv3
        output2 = layers.SeparableConv2D(filters = 128, kernel_size = (1, 8),
                                         data_format='channels_last',
                                         strides = (1, 1), padding = "SAME",
                                         use_bias = False, kernel_initializer = weight_initializer,
                                         bias_initializer = bias_initializer, name = "large_cnn_conv_3")(residual2) 
        print("8output2.shape", output2.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        output2 = layers.BatchNormalization(name = "large_cnn_batchnorm_3")(output2)        #全是默认参数
        print("9output2.shape", output2.shape)
        output2 = layers.ReLU(name = "large_cnn_relu_3")(output2)                    #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        print("10output2.shape", output2.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        
        #conv4
        output2 = layers.SeparableConv2D(filters = 128, kernel_size = (1,8),
                                         data_format='channels_last',
                                         strides = 1, padding = "SAME",
                                         use_bias = False, kernel_initializer = weight_initializer,
                                         bias_initializer = bias_initializer, name = "large_cnn_conv_4")(output2)
        print("11output2.shape", output2.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        output2 = layers.BatchNormalization(name = "large_cnn_batchnorm_4")(output2)        #全是默认参数
        print("12output2.shape", output2.shape)
        
        #residual
        output2 = layers.Add(name = "residual_add2")([output2, residual2])
        print("shape after add residual2:", output2.shape)
        
        output2 = layers.ReLU(name = "large_cnn_relu_4")(output2)                    #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        print("13output2.shape", output2.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        
        
        output2 = layers.MaxPool2D(pool_size = (2, 1), strides = (2, 1),          #MaxPooling2D和MaxPool2D都可以
                                   padding = "SAME", name = "large_cnn_maxpool_2",
                                   data_format = 'channels_first')(output2)
        print("14output2.shape", output2.shape)                                #(None, 20, 8, 1, 128)——>(None, 20, 8, 128)
        
        output2 = layers.Reshape((output2.shape[1], -1))(output2)
        print("15output2.shape", output2.shape)                                #(None, 20, 1024)
        
        cnn_outputs.append(output2)
        
        
        
        
        ####子卷积网络2    
        #conv1
        output3 = layers.Conv1D(filters = 64, kernel_size = 800,
                                strides = 100, padding = "SAME",
                                use_bias = False, kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer, name = "middle_cnn_conv_1",
                                input_shape = (3000,1))(input)
        print("1output3.shape", output3.shape)                                #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        output3 = layers.BatchNormalization(name = "middle_cnn_batchnorm_1")(output3)       #全是默认参数
        print("2output3.shape", output3.shape)                                 #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        output3 = layers.ReLU(name = "middle_cnn_relu_1")(output3)                    #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        print("3output3.shape", output3.shape)                                 #(None, 20, 60, 1, 64)——>(None, 20, 60, 64)
        
        output3 = layers.MaxPool2D(pool_size = (2, 1), strides = (2, 1),           #MaxPooling2D和MaxPool2D都可以
                                   padding = "SAME", name = "middle_cnn_maxpool_1",
                                   data_format = 'channels_first')(output3)

        print("4output3.shape", output3.shape)                                 #(None, 20, 15, 1, 64)——>(None, 20, 15, 64)
        output3 = layers.Dropout(rate = 0.5, name = "middle_cnn_dropout")(output3)
        
        #conv2
        output3 = layers.Conv1D(filters = 128, kernel_size = 8,
                                strides = 1, padding = "SAME",
                                use_bias = False, kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer, name = "middle_cnn_conv_2")(output3)
        print("5output3.shape", output3.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        output3 = layers.BatchNormalization(name = "middle_cnn_batchnorm_2")(output3)        #全是默认参数
        print("6output3.shape", output3.shape)
        
        #residual
        residual3 = layers.ReLU(name = "middle_cnn_relu_2")(output3)                    #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        print("7residual3.shape", residual3.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        
        
        #conv3
        output3 = layers.SeparableConv2D(filters = 128, kernel_size = (1, 8),
                                         data_format='channels_last',
                                         strides = (1, 1), padding = "SAME",
                                         use_bias = False, kernel_initializer = weight_initializer,
                                         bias_initializer = bias_initializer, name = "middle_cnn_conv_3")(residual3) 
        print("8output3.shape", output3.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        output3 = layers.BatchNormalization(name = "middle_cnn_batchnorm_3")(output3)        #全是默认参数
        print("9output3.shape", output3.shape)
        output3 = layers.ReLU(name = "middle_cnn_relu_3")(output3)                    #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        print("10output3.shape", output3.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        
        #conv4
        output3 = layers.SeparableConv2D(filters = 128, kernel_size = (1,8),
                                         data_format='channels_last',
                                         strides = 1, padding = "SAME",
                                         use_bias = False, kernel_initializer = weight_initializer,
                                         bias_initializer = bias_initializer, name = "middle_cnn_conv_4")(output3)
        print("11output3.shape", output3.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        output3 = layers.BatchNormalization(name = "middle_cnn_batchnorm_4")(output3)        #全是默认参数
        print("12output3.shape", output3.shape)
        
        #residual
        output3 = layers.Add(name = "residual_add3")([output3, residual3])
        print("shape after add residual2:", output3.shape)
        
        output3 = layers.ReLU(name = "middle_cnn_relu_4")(output3)                    #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        print("13output3.shape", output3.shape)                                 #(None, 20, 15, 1, 128)——>(None, 20, 15, 128)
        
        output3 = layers.MaxPool2D(pool_size = (2, 1), strides = (2, 1),          #MaxPooling2D和MaxPool2D都可以
                                   padding = "SAME", name = "middle_cnn_maxpool_2",
                                   data_format = 'channels_first')(output3)
        print("14output3.shape", output3.shape)                                #(None, 20, 8, 1, 128)——>(None, 20, 8, 128)
        
        output3 = layers.Reshape((output3.shape[1], -1))(output3)
        print("15output3.shape", output3.shape)                                #(None, 20, 1024)
        
        cnn_outputs.append(output3)
        
        
        
        
        cnn_output = layers.Concatenate()(cnn_outputs)
        cnn_output = layers.Dropout(rate = 0.5, name = "cnn_output_dropout")(cnn_output)
        
        print("final_cnn_output.shape", cnn_output.shape)         #(None, 20, 3072)


        #BiLSTM
        forward_layer = layers.LSTM(128, return_sequences = True)
        backward_layer = layers.LSTM(128, return_sequences = True,      #, activation='relu'
                                     go_backwards = True)
        final_output2 = layers.Bidirectional(forward_layer, backward_layer = backward_layer, merge_mode = 'concat',
                                             input_shape = (20, 4096))(cnn_output)
        final_output2 = layers.Dropout(rate = 0.5, name = "bi_lstm_1_dropout")(final_output2)
        print("final_output2.shape", final_output2.shape)          #(None, 20, 1024)


        #dense部分
        final_output = layers.Dense(units = 5, use_bias = True,
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = tf.constant_initializer(0.0),
                                    name = "final_fc")(final_output2)
        
        final_output = layers.Softmax(name = "final_softmax")(final_output)
        
    
        return [input], [final_output]


