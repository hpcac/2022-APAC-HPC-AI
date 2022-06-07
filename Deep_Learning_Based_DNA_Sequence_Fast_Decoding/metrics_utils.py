import numpy as np
import csv
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + smooth) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def save_result(result_dic, path=''):
    with open(path + "evaluation_metrics.csv", "w") as file:
        writer = csv.DictWriter(file, result_dic.keys())
        writer.writeheader()
        writer.writerow(result_dic)
        