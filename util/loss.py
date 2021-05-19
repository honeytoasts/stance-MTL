# 3rd-party module
from torch import nn

def loss_function(task_id: int,
                  predict,
                  target,
                  nli_loss_weight: float):

    # get cross entropy Loss
    ce_loss = nn.CrossEntropyLoss()
    task_loss = ce_loss(predict, target)

    # get final loss, for NLI: multiply weight
    loss = (task_loss if task_id == 0 else
            task_loss * nli_loss_weight)

    return loss