# 3rd-party module
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# self-made module
from util import loss
from util import scorer

def evaluate_function(model, config, batch_iterator,
                      device, phase='train', task='stance'):
    total_loss, total_lexicon_loss = 0.0, 0.0
    stance_loss, stance_lexicon_loss = 0.0, 0.0
    nli_loss, nli_lexicon_loss = 0.0, 0.0
    stance_batch_count, nli_batch_count = 0, 0
    all_label_stance_y, all_pred_stance_y = [], []
    all_label_nli_y, all_pred_nli_y = [], []

    model.eval()
    with torch.no_grad():
        for task_id, x1, x2, lexicon, y in batch_iterator:
            # specify device for data
            x1 = x1.to(device)
            x2 = x2.to(device)
            lexicon = lexicon.to(device)
            y = y.to(device)

            # get predict label and attention weight
            pred_y, attn_weight = model(task_id, x1, x2)

            # get batch loss
            batch_loss, batch_lexicon_loss = \
                loss.loss_function(task_id=task_id,
                                   lexicon_vector=lexicon,
                                   predict=pred_y,
                                   target=y,
                                   attn_weight=attn_weight,
                                   nli_loss_weight=config.nli_loss_weight,
                                   lexicon_loss_weight=config.lexicon_loss_weight)

            # sum the batch loss
            total_loss += batch_loss
            total_lexicon_loss += batch_lexicon_loss

            if task_id == 0:  # stance detection
                stance_loss += batch_loss
                stance_lexicon_loss += batch_lexicon_loss
                stance_batch_count += 1

                all_label_stance_y.extend(y.tolist())
                all_pred_stance_y.extend(torch.argmax(pred_y, axis=1).cpu().tolist())
            elif task_id == 1:  # NLI
                nli_loss += batch_loss
                nli_lexicon_loss += batch_lexicon_loss
                nli_batch_count += 1

                all_label_nli_y.extend(y.tolist())
                all_pred_nli_y.extend(torch.argmax(pred_y, axis=1).cpu().tolist())

    # evaluate loss
    total_loss = total_loss / len(batch_iterator)
    total_lexicon_loss = total_lexicon_loss / len(batch_iterator)
    if phase == 'train':
        stance_loss = stance_loss / stance_batch_count
        stance_lexicon_loss = stance_lexicon_loss / stance_batch_count
        nli_loss = nli_loss / nli_batch_count
        nli_lexicon_loss = nli_lexicon_loss / nli_batch_count

    # evaluate accuracy
    if config.stance_dataset == 'semeval2016' and task == 'stance':
        # semeval2016 benchmark just consider f1 score for "favor (0)" and "against (1)" label
        acc = f1_score(all_label_stance_y,
                       all_pred_stance_y,
                       average='macro',
                       labels=[0, 1],
                       zero_division=0)
    elif config.stance_dataset == 'fnc-1' and task == 'stance':
        acc = scorer.fnc_score(all_label_stance_y, all_pred_stance_y)
    else:
        acc = accuracy_score(all_label_nli_y, all_pred_nli_y) 

    if phase == 'train':
        return (total_loss, total_lexicon_loss,
                stance_loss, stance_lexicon_loss,
                nli_loss, nli_lexicon_loss,
                acc)
    elif phase == 'valid':
        return (total_loss, total_lexicon_loss, acc)