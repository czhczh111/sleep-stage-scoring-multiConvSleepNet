from tensorflow.keras import layers, models
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import tensorflow as tf

class baseCNN(models.Model):
    def __init__(self, padding: str ='same', build: bool = True, **kwargs):
        super(baseCNN, self).__init__()

        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_length']          #3000
        if build:
            super().__init__(*self.init_model())

    def init_model(self, input: KerasTensor = None) -> (list, list):
        if input is None:
            input = layers.Input(shape=(1, self.sleep_epoch_length, 1))                  ##[B, 3000, 1, 1]
        print("input.shape",input.shape)                                              #(None, 3000, 1, 1)
        
        weight_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="normal")
        bias_initializer = tf.zeros_initializer()
        output = layers.Conv1D(filters = 128, kernel_size = 50,
                               strides = 6, padding = "SAME",
                               use_bias = False, kernel_initializer = weight_initializer,
                               bias_initializer = bias_initializer, name = "cnn_conv_1",
                               input_shape = (3000,1))(input)

        
        print("1output.shape", output.shape)     #(None, 500, 1, 128)
        output = layers.BatchNormalization(momentum = 0.99, epsilon = 0.001, center = True, scale = True,
                                           beta_initializer = tf.zeros_initializer(),
                                           gamma_initializer = tf.ones_initializer(),
                                           moving_mean_initializer = tf.zeros_initializer(),
                                           moving_variance_initializer = tf.ones_initializer(),
                                           name = "cnn_batchnorm_1")(output)
        #output = layers.BatchNormalization(name = "cnn_batchnorm_1")
        print("2output.shape", output.shape)
        
        output = layers.ReLU(name = "cnn_relu_1")(output)
        print("3output.shape", output.shape)
        output = layers.MaxPooling2D(pool_size = (8, 1), strides = (8, 1),
                                     padding = "SAME", name = "cnn_maxpool_1",
                                     data_format = 'channels_first')(output)
        print("4output.shape", output.shape)
        output = layers.Dropout(rate = 0.5, name="cnn_dropout_1")(output)
        print("5output.shape", output.shape)
        
#         output = layers.Conv1D(filters = 128, kernel_size = 8,
#                                strides = 1, padding = "SAME",
#                                use_bias = False, kernel_initializer = weight_initializer,
#                                bias_initializer = bias_initializer, name = "cnn_conv_2")(output)
#         print("6output.shape", output.shape)
#         output = layers.BatchNormalization(name = "cnn_batchnorm_2")(output)
#         print("7output.shape", output.shape)
#         output = layers.ReLU(name = "cnn_relu_2")(output)
#         print("8output.shape", output.shape)
        
#         output = layers.Conv1D(filters = 128, kernel_size = 8,
#                                strides = 1, padding = "SAME",
#                                use_bias = False, kernel_initializer = weight_initializer,
#                                bias_initializer = bias_initializer, name = "cnn_conv_3")(output)
#         print("9output.shape", output.shape)
#         output = layers.BatchNormalization(name = "cnn_batchnorm_3")(output)
#         print("10output.shape", output.shape)
#         output = layers.ReLU(name = "cnn_relu_3")(output)
#         print("11output.shape", output.shape)
        
        output = layers.Conv1D(filters = 128, kernel_size = 8,
                               strides = 1, padding = "SAME",
                               use_bias = False, kernel_initializer = weight_initializer,
                               bias_initializer = bias_initializer, name = "cnn_conv_4")(output)
        print("12output.shape", output.shape)
        output = layers.BatchNormalization(name = "cnn_batchnorm_4")(output)
        print("13output.shape", output.shape)                                    #[B,1,63,128]
        output = layers.ReLU(name = "cnn_relu_4")(output)
        print("14output.shape", output.shape)

        output = layers.MaxPool2D(pool_size = (4, 1), strides = (4, 1),
                                  padding = "SAME", name = "cnn_maxpool_2",
                                  data_format = 'channels_first')(output)
        print("15output.shape", output.shape)
        
        output = layers.Flatten(name = "cnn_flatten_1")(output)
        print("16output.shape", output.shape)
        output = layers.Dropout(rate = 0.5, name="cnn_dropout_2")(output)
        print("17output.shape", output.shape)
        
        
        output = tf.keras.layers.Dense(units = 5, use_bias = True,
                                       kernel_initializer = weight_initializer,
                                       bias_initializer = tf.constant_initializer(0.0),
                                       name = "cnn_fc")(output)

        output = layers.Softmax(name = "cnn_softmax")(output)

        return [input], [output]
