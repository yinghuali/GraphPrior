import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler


def Margin_score(x):
    output_sort = np.sort(x)
    margin_score = output_sort[:, -1] - output_sort[:, -2]
    return margin_score


def DeepGini_score(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    return gini_score


def Variance_score(x):
    var = np.var(x, axis=1)
    return var


def LeastConfidence_score(x):
    max_pre = x.max(1)
    return max_pre


def get_uncertainty_feature(prediction):
    df = pd.DataFrame(columns=['margin'])
    df['margin'] = Margin_score(prediction)
    df['deepgini'] = DeepGini_score(prediction)
    df['variance'] = Variance_score(prediction)
    df['least'] = LeastConfidence_score(prediction)
    return df

