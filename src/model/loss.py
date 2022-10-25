from pdb import set_trace as bp
import torch.nn.functional as F
import torch



def weighted_bce_loss(output, target, w_p=0.99, w_n=0.01, epsilon=1e-7):
    loss_pos = -1 * torch.mean(w_p * target * torch.log(output + epsilon))
    loss_neg = -1 * torch.mean(w_n * (1-target) * torch.log((1-output) + epsilon))
    loss = loss_pos + loss_neg
    return loss

