import numpy as np
import tensorflow.keras.backend as K

def weighted_categorical_cross_entropy(weights: np.ndarray):

    weights = K.variable(weights)

    def loss_fn(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip 防止NaN's和Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss_fn
