# 3rd-party module
import torch
from torch import nn

def loss_function(task_id,
                  predict, target,
                  lexicon_vector,
                  task_weight, shared_weight,
                  nli_loss_weight, lexicon_loss_weight):

    # get cross entropy Loss
    ce_loss = nn.CrossEntropyLoss()
    task_loss = ce_loss(predict, target)

    # get attention weight
    sum_weight = task_weight + shared_weight
    norm_weight = nn.functional.normalize(sum_weight)
    # norm_weight = sum_weight / sum_weight.max(dim=1, keepdim=True)[0]

    # get MSE loss (lexicon loss)
    mse_loss = nn.MSELoss()
    lexicon_loss = mse_loss(norm_weight, lexicon_vector)

    # get final loss
    # for stance detection: add lexicon loss
    # for NLI: multiply weight, may or may not add lexicon loss
    if task_id == 0:
        total_loss = task_loss + lexicon_loss_weight * lexicon_loss
    elif task_id == 1:
        total_loss = nli_loss_weight * task_loss
        # total_loss = nli_loss_weight * task_loss + \
        #              lexicon_loss_weight * lexicon_loss

    return total_loss, lexicon_loss