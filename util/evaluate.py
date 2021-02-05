# 3rd-party module
import torch
import pandas as pd

# self-made module
from util import loss
from util import scorer

def evaluate_function(device, model, config, batch_iterator):
    total_loss, total_lexicon_loss = 0.0, 0.0
    stance_loss, stance_lexicon_loss = 0.0, 0.0
    nli_loss, nli_lexicon_loss = 0.0, 0.0

    stance_batch_count, nli_batch_count = 0, 0
    all_stance_target = []
    all_stance_label, all_nli_label = [], []
    all_stance_pred, all_nli_pred = [], []

    # evaluate model
    model.eval()
    with torch.no_grad():
        for task_id, target_name, \
            x1, x2, lexicon, y in batch_iterator:
            # specify device for data
            x1 = x1.to(device)
            x2 = x2.to(device)
            lexicon = lexicon.to(device)
            y = y.to(device)

            # get predict label and attention weight
            pred_y, task_weight, shared_weight = \
                model(task_id, x1, x2)

            # calculate loss
            batch_loss, batch_lexicon_loss = \
                loss.loss_function(
                    task_id=task_id,
                    predict=pred_y,
                    target=y,
                    lexicon_vector=lexicon,
                    task_weight=task_weight,
                    shared_weight=shared_weight,
                    nli_loss_weight=config.nli_loss_weight,
                    lexicon_loss_weight=config.lexicon_loss_weight)

            # sum the batch loss
            total_loss += batch_loss
            total_lexicon_loss += batch_lexicon_loss

            # get target, label and predict
            if task_id == 0:  # for stance detection
                stance_loss += batch_loss
                stance_lexicon_loss += batch_lexicon_loss
                stance_batch_count += 1

                all_stance_target.extend(target_name)
                all_stance_pred.extend(
                    torch.argmax(pred_y, axis=1).cpu().tolist())
                all_stance_label.extend(y.tolist())
            elif task_id == 1:  # for NLI
                nli_loss += batch_loss
                nli_lexicon_loss += batch_lexicon_loss
                nli_batch_count += 1

                all_nli_pred.extend(
                    torch.argmax(pred_y, axis=1).cpu().tolist())
                all_nli_label.extend(y.tolist())

    # check batch count
    assert len(batch_iterator) == \
        stance_batch_count + nli_batch_count

    # evaluate loss
    total_loss = total_loss / len(batch_iterator)
    total_lexicon_loss = total_lexicon_loss / len(batch_iterator)
    stance_loss = stance_loss / stance_batch_count
    stance_lexicon_loss = stance_lexicon_loss / stance_batch_count
    nli_loss = nli_loss / nli_batch_count
    nli_lexicon_loss = nli_lexicon_loss / nli_batch_count

    # evaluate accuracy
    if config.stance_dataset == 'semeval2016':
        stance_score = \
            scorer.semeval_score(targets=pd.Series(all_stance_target),
                                 label_y=all_stance_label,
                                 pred_y=all_stance_pred)
    elif config.stance_dataset == 'fnc-1':
        stance_score = \
            scorer.fnc_score(label_y=all_stance_label,
                             pred_y=all_stance_pred)
    nli_score = scorer.nli_score(label_y=all_nli_label,
                                 pred_y=all_nli_pred)

    return (total_loss.item(), total_lexicon_loss.item(),
            stance_loss.item(), stance_lexicon_loss.item(),
            nli_loss.item(), nli_lexicon_loss.item(),
            stance_score, nli_score)

def stance_evaluate_function(device, model, config, batch_iterator):
    total_loss, total_lexicon_loss = 0.0, 0.0
    all_target = []
    all_label, all_pred = [], []

    # evaluate model
    model.eval()
    with torch.no_grad():
        for task_id, target_name, \
            x1, x2, lexicon, y in batch_iterator:
            # specify device for data
            x1 = x1.to(device)
            x2 = x2.to(device)
            lexicon = lexicon.to(device)
            y = y.to(device)

            # get predict label and attention weight
            pred_y, task_weight, shared_weight = \
                model(task_id, x1, x2)

            # calculate loss
            batch_loss, batch_lexicon_loss = \
                loss.loss_function(
                    task_id=task_id,
                    predict=pred_y,
                    target=y,
                    lexicon_vector=lexicon,
                    task_weight=task_weight,
                    shared_weight=shared_weight,
                    nli_loss_weight=config.nli_loss_weight,
                    lexicon_loss_weight=config.lexicon_loss_weight)

            # sum the batch loss
            total_loss += batch_loss
            total_lexicon_loss += batch_lexicon_loss

            # get target, label and predict
            all_target.extend(target_name)
            all_pred.extend(
                torch.argmax(pred_y, axis=1).cpu().tolist())
            all_label.extend(y.tolist())

    # evaluate loss
    total_loss = total_loss / len(batch_iterator)
    total_lexicon_loss = total_lexicon_loss / len(batch_iterator)

    # evaluate accuracy
    if config.stance_dataset == 'semeval2016':
        stance_score = \
            scorer.semeval_score(targets=all_target,
                                 label_y=all_label,
                                 pred_y=all_pred)
    elif config.stance_dataset == 'fnc-1':
        stance_score = \
            scorer.fnc_score(label_y=all_label,
                             pred_y=all_pred)

    return (total_loss.item(), total_lexicon_loss.item(),
            stance_score)