# 3rd-party module
import pandas as pd
import sklearn
from sklearn import metrics

def semeval_score(targets, label_y, pred_y):
    # only consider "favor (0)" and "against (1)"
    consider_labels = [0, 1]
    target_name = ['atheism', 'climate change is a real concern',
                   'feminist movement', 'hillary clinton',
                   'legalization of abortion']

    target_series, label_series, pred_series = \
        pd.Series(targets), pd.Series(label_y), pd.Series(pred_y)
    target_f1 = []

    # get f1-score for each target
    for target in target_name:
        labels = label_series[target_series == target].tolist()
        preds = pred_series[target_series == target].tolist()
        f1 = metrics.f1_score(labels, preds,
                              average='macro', labels=consider_labels)

        target_f1.append(f1.item())

    # get macro-f1 and micro-f1
    macro_f1 = sum(target_f1) / len(target_f1)
    micro_f1 = metrics.f1_score(label_y,
                                pred_y,
                                average='macro',
                                labels=consider_labels,
                                zero_division=0)

    return (target_f1, macro_f1, micro_f1.item())

def fnc_score(label_y, pred_y):
    # specfiy related label
    related = [0, 1, 2]
    unrelated = 3

    # calculate score
    score = 0.0
    for label, pred in zip(label_y, pred_y):
        if label == pred:
            score += 0.25
            if label != unrelated:
                score += 0.50
        if label in related and pred in related:
            score += 0.25

    score = score / len(label_y)

    return score

def nli_score(label_y, pred_y):
    acc = metrics.accuracy_score(label_y, pred_y)

    return acc.item()