# 3rd-party module
import torch

def loss_function(lexicon_vector, tokenizer,
                  predict, target, attn_weight, beta, device):
    # calculate cross entropy Loss
    ce_loss = torch.nn.CrossEntropyLoss()
    predict, target = predict.to(device), target.to(device)
    main_loss = ce_loss(predict, target)

    # calculate lexicon loss
    mse_loss = torch.nn.MSELoss()
    attn_weight = attn_weight.to(device)
    lexicon_vector = lexicon_vector.to(device)
    lexicon_loss = mse_loss(attn_weight, lexicon_vector)

    # get final loss
    loss = main_loss + beta*lexicon_loss

    return loss