import numpy as np
import tensorflow.keras.backend as K


def weighted_categorical_cross_entropy(weights: np.ndarray):
    """
    加权的keras.objectives.categorical_crossentropy
    weights: 每个类别权重，list
    """

    weights = K.variable(weights)

    def loss_fn(y_true, y_pred):
        #归一化（不用softmax）
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip 防止NaN，Inf
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss_fn
