# 3rd-party module
import pandas as pd
from sklearn import metrics

def stance_score(targets, labels, predicts):
    # only consider "favor (0)" and "against (1)"
    consider_labels = [0, 1]
    target_name = ['atheism', 'climate change is a real concern',
                   'feminist movement', 'hillary clinton',
                   'legalization of abortion']

    # initialize series
    target_series, label_series, pred_series = (
        pd.Series(targets), pd.Series(labels), pd.Series(predicts))

    # get f1-score for each target
    target_f1 = []
    for target in target_name:
        label = label_series[target_series == target].tolist()
        predict = pred_series[target_series == target].tolist()

        f1 = metrics.f1_score(label,
                              predict,
                              average='macro',
                              labels=consider_labels,
                              zero_division=0)
        target_f1.append(f1.item())

    # get macro-f1 and micro-f1
    macro_f1 = sum(target_f1) / len(target_f1)
    micro_f1 = metrics.f1_score(labels,
                                predicts,
                                average='macro',
                                labels=consider_labels,
                                zero_division=0)

    return target_f1, macro_f1, micro_f1.item()

def nli_score(labels, predicts):
    acc = metrics.accuracy_score(labels, predicts)

    return acc.item()