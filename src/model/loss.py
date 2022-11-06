from pdb import set_trace as bp
import torch.nn.functional as F
import torch



def weighted_bce_loss(output, target, w_p=99, w_n=1, epsilon=1e-7):
    loss_pos = -1 * torch.mean(w_p * target * torch.log(output + epsilon))
    loss_neg = -1 * torch.mean(w_n * (1-target) * torch.log((1-output) + epsilon))
    loss = loss_pos + loss_neg
    return loss


def cost_sensetive_bce_loss(output, target, epsilon=1e-7, w_tp=99, w_tn=0, w_fp=1, w_fn=99):

    fn = w_fn * torch.mean(target * torch.log(output+epsilon))
    tp = w_tp * torch.mean(target * torch.log((1-output)+epsilon))
    fp = w_fp * torch.mean((1-target) * torch.log((1-output)+epsilon))
    tn = w_tn * torch.mean((1-target) * torch.log(output+epsilon))
    return -(fn+tp+fp+tn)