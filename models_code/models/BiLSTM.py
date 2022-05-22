from tensorflow.keras import layers, models
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import tensorflow as tf


class BiLSTM(models.Model):
    def __init__(self, padding: str ='same', build: bool = True, **kwargs):
        super(BiLSTM, self).__init__()

        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_length']          #3000
        self.sequence_length = kwargs['sequence_length']                #20

        if build:
            super().__init__(*self.init_model())

    def init_model(self, input: KerasTensor = None) -> (list, list):
        
        cnn_outputs = []
        if input is None:
            input = layers.Input(shape=(20, self.sleep_epoch_length))  ##[20, 3000, 1, 1]
        print("---input.shape",input.shape)                         #(None, 20, 3000, 1, 1)
        

        #Bilstm
        forward_layer = layers.LSTM(512, return_sequences = True)
        backward_layer = layers.LSTM(512, return_sequences = True,      #, activation='relu'
                       go_backwards = True)
        final_output2 = layers.Bidirectional(forward_layer, backward_layer = backward_layer, merge_mode = 'concat',
                                 input_shape = (20, 3000))(input)
        final_output2 = layers.Dropout(rate = 0.5, name = "bi_lstm_1_dropout")(final_output2)
        print("final_output2.shape", final_output2.shape)          #(None, 20, 1024)

        #dense
        weight_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="normal")
        bias_initializer = tf.zeros_initializer()
        
        final_output = layers.Dense(units = 5, use_bias = True,
                             kernel_initializer = weight_initializer,
                             bias_initializer = tf.constant_initializer(0.0),
                             name = "final_fc")(final_output2)
        
        print("final_output.shape", final_output.shape)               #(None, 20, 5)
        final_output = layers.Softmax(name = "final_softmax")(final_output)
        
    
        return [input], [final_output]
