# 3rd-party module
import torch

def loss_function(lexicon_vector, tokenizer,
                  predict, target, attn_weight,
                  lexicon_loss_weight, device):
    # get cross entropy Loss
    ce_loss = torch.nn.CrossEntropyLoss()
    predict, target = predict.to(device), target.to(device)
    total_loss = ce_loss(predict, target)

    # get lexicon loss if beta != 0
    if lexicon_loss_weight != 0:
        # get attention weight normalization
        norm_weight = torch.nn.functional.normalize(attn_weight)

        # calculate lexicon loss
        mse_loss = torch.nn.MSELoss()
        norm_weight = norm_weight.to(device)
        lexicon_vector = lexicon_vector.to(device)
        lexicon_loss = mse_loss(norm_weight, lexicon_vector)

    # get final loss
    if lexicon_loss_weight != 0:
        lexicon_loss = lexicon_loss_weight * lexicon_loss
        total_loss = total_loss + lexicon_loss

        return total_loss, lexicon_loss
    else:
        return total_loss, torch.tensor(0.0)  # no lexicon loss