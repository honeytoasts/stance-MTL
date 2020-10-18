# 3rd-party module
import pandas as pd
import sklearn
from sklearn import metrics

def score_function(dataset, label_y, pred_y, targets=None):
    # specify label according dataset
    if dataset == 'semeval2016':
        labels = [0, 1, 2]
    elif dataset == 'fnc-1':
        labels = [0, 1, 2, 3]

    # get confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true=label_y,
                                                y_pred=pred_y,
                                                labels=labels)

    # get classification report
    labels_str = [str(label) for label in labels]
    report = metrics.classification_report(y_true=label_y,
                                           y_pred=pred_y,
                                           target_names=labels_str,
                                           output_dict=True)

    # get precision, recall, f1-score for each "label"
    label_precision = [report[label]['precision'] for label in labels_str]
    label_recall = [report[label]['recall'] for label in labels_str]
    label_f1 = [report[label]['f1-score'] for label in labels_str]

    # With SemEval2016 dataset, get f1 score for each "target"
    if dataset == 'semeval2016':
        target_name = ['atheism', 'climate change is a real concern',
                       'feminist movement', 'hillary clinton',
                       'legalization of abortion']
        label_series, pred_series = pd.Series(label_y), pd.Series(pred_y)
        target_f1 = []

        # get f1-score for each target
        for target in target_name:
            labels = label_series[targets == target].tolist()
            preds = pred_series[targets == target].tolist()
            f1 = metrics.f1_score(labels, preds, average='macro', labels=[0, 1])

            target_f1.append(f1)

        # get macro-f1 and micro-f1
        macro_f1 = sum(target_f1) / len(target_f1)
        micro_f1 = metrics.f1_score(label_y, pred_y, average='macro', labels=[0, 1])

        return (confusion_matrix,
                label_precision, label_recall, label_f1,
                target_f1, macro_f1, micro_f1)

    elif dataset == 'fnc-1':
        return (confusion_matrix,
                label_precision, label_recall, label_f1)