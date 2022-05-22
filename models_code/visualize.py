import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          normalize: bool = True, title: str = 'Confusion matrix',
                          color: str = 'Blues', path: str = ''):
    """
    绘制混淆矩阵
    cm: 混淆矩阵
    classes: 类名列表
    color: 颜色
    """

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap = color)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = np.max(cm) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(path+title+'.png')

