# 3rd-party module
import torch

def loss_function(task_id, lexicon_vector,
                  predict, target, attn_weight,
                  nli_loss_weight, lexicon_loss_weight):

    # get cross entropy Loss
    ce_loss = torch.nn.CrossEntropyLoss()
    total_loss = ce_loss(predict, target)

    # get lexicon loss
    if lexicon_loss_weight != 0:
        # get attention weight normalization
        task_weight, shared_weight = attn_weight
        sum_weight = task_weight + shared_weight
        norm_weight = torch.nn.functional.normalize(sum_weight)

        # calculate MSE loss
        mse_loss = torch.nn.MSELoss()
        lexicon_loss = mse_loss(norm_weight, lexicon_vector)

    # get final loss
    if lexicon_loss_weight != 0:
        lexicon_loss = lexicon_loss_weight * lexicon_loss
        total_loss = total_loss + lexicon_loss

        if task_id == 1:  # multiply nli loss weight
            total_loss = nli_loss_weight * total_loss
            lexicon_loss = nli_loss_weight * lexicon_loss

        return total_loss, lexicon_loss
    else:
        if task_id == 1:  # multiply nli loss weight
            total_loss = nli_loss_weight * total_loss

        return total_loss, torch.tensor(0.0)  # no lexicon loss