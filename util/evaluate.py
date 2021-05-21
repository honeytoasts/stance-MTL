# 3rd-party module
import argparse
import torch

# self-made module
from util import model
from util import loss
from util import scorer

@torch.no_grad()
def evaluate_function(device: torch.device,
                      model,
                      config: argparse.Namespace,
                      batch_iterator,
                      evaluate_nli=True):

    total_loss, stance_loss, nli_loss = 0.0, 0.0, 0.0

    all_stance_target = []
    all_stance_label, all_nli_label = [], []
    all_stance_pred, all_nli_pred = [], []

    # evaluate model
    model.eval()
    for (task_id, target_name,
         task_target, shared_target,
         task_claim, shared_claim,
         task_attn_mask, shared_attn_mask,
         task_adj_matrix, shared_adj_matrix,
         label) in batch_iterator:

        # specify device for data
        task_target = task_target.to(device)
        shared_target = shared_target.to(device)
        task_claim = task_claim.to(device)
        shared_claim = shared_claim.to(device)
        task_attn_mask = task_attn_mask.to(device)
        shared_attn_mask = shared_attn_mask.to(device)
        task_adj_matrix = task_adj_matrix.to(device)
        shared_adj_matrix = shared_adj_matrix.to(device)
        label = label.to(device)

        # get predict label
        predict, _ = model(task_id,
                           task_target, shared_target,
                           task_claim, shared_claim,
                           task_attn_mask, shared_attn_mask,
                           task_adj_matrix, shared_adj_matrix)

        # calculate loss
        batch_loss = (
            loss.loss_function(task_id=task_id,
                               predict=predict,
                               target=label,
                               nli_loss_weight=config.nli_loss_weight))

        # sum the batch loss
        total_loss += batch_loss * len(label)
        if task_id == 0:
            stance_loss += batch_loss * len(label)
        elif task_id == 1:
            nli_loss += batch_loss * len(label)

        # get target, labeland predict
        if task_id == 0:
            all_stance_target.extend(target_name)
            all_stance_pred.extend(
                torch.argmax(predict, axis=1).cpu().tolist())
            all_stance_label.extend(label.cpu().tolist())
        elif task_id == 1:
            all_nli_pred.extend(
                torch.argmax(predict, axis=1).cpu().tolist())
            all_nli_label.extend(label.cpu().tolist())

    # evaluate loss
    total_loss /= (len(all_stance_label)+len(all_nli_label))
    stance_loss /= len(all_stance_label)
    if evaluate_nli:
        nli_loss /= len(all_nli_label)
    else:
        nli_loss = torch.tensor(0.0)

    # evaluate f1 score
    target_f1, macro_f1, micro_f1 = (
        scorer.stance_score(targets=all_stance_target,
                            labels=all_stance_label,
                            predicts=all_stance_pred))
    if evaluate_nli:
        nli_acc = (
            scorer.nli_score(labels=all_nli_label,
                             predicts=all_nli_pred))
    else:
        nli_acc = 0.0

    return (total_loss.item(), stance_loss.item(), nli_loss.item(),
            target_f1, macro_f1, micro_f1,
            nli_acc)