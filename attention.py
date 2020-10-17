import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MyAttention(nn.Module):
    def __init__(self, opt):
        super(MyAttention, self).__init__()
        self.device = opt.device

    def forward(self, generated, known, mask):
        return MyAttentionFunction.apply(generated, known, mask, self.device)


class MyAttentionFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, generated, known, mask, device):
        k = 2
        ctx.k = k

        ctx.flag = F.interpolate(mask.clone(), (32, 32)).view(1, -1).to(device)

        patches_all = MyUnfold(generated, 1, 1)
        patches = patches_all[ctx.flag == 1]
        patches = patches.view(1, patches.size(0), -1)

        known_patches_all = MyUnfold(known, 1, 1)
        known_patches = known_patches_all[ctx.flag == 0]
        known_patches = known_patches.view(1, known_patches.size(0), -1)

        num = torch.einsum('bik,bjk->bij', [patches, known_patches])
        norm_known = torch.einsum("bij,bij->bi", [known_patches, known_patches])
        norm_patches = torch.einsum("bij,bij->bi", [patches, patches])
        den = torch.sqrt(torch.einsum('bi,bj->bij', [norm_patches, norm_known]))

        num1 = torch.einsum('bik,bjk->bij', [patches, patches])
        den1 = torch.sqrt(torch.einsum('bi,bj->bij', [norm_patches, norm_patches]))

        cosine0 = num / den
        cosine1 = num1 / den1
        weight0, indexes0 = torch.topk(cosine0, k, dim=2)
        weight1, indexes1 = torch.topk(cosine1, k, dim=2)
        ctx.weights = []
        for i in range(k):
            ctx.weights.append(weight0[0, :, i].mean().item())
        for i in range(k):
            ctx.weights.append(weight1[0, :, i].mean().item())
        ctx.weights = np.exp(ctx.weights) / np.sum(np.exp(ctx.weights))

        mask_indexes = (ctx.flag == 1).nonzero()
        non_mask_indexes = []
        for i in range(k):
            non_mask_indexes.append((ctx.flag == 0).nonzero()[indexes0[:, :, i]])
        for i in range(k):
            non_mask_indexes.append((ctx.flag == 1).nonzero()[indexes1[:, :, i]])

        ctx.ind = []
        for i in range(k * 2):
            ind_lst = torch.Tensor(1, 1024, 1024).zero_().to(device)
            ind_lst[0][mask_indexes, non_mask_indexes[i]] = 1
            ctx.ind.append(ind_lst)
            if i == 0:
                rtn = torch.bmm(ind_lst, known_patches_all) * ctx.weights[i]
            elif i < k:
                rtn += torch.bmm(ind_lst, known_patches_all) * ctx.weights[i]
            else:
                rtn += torch.bmm(ind_lst, patches_all) * ctx.weights[i]

        rtn = rtn.view(1, 32, 32, known.size()[1]).permute(0, 3, 1, 2)
        return torch.cat([generated, known, rtn], 1)

    @staticmethod
    def backward(ctx, grad_output):

        c = grad_output.size(1)

        grad_former_all = grad_output[:, 0:c//3, :, :]
        grad_latter_all = grad_output[:, c//3: c*2//3, :, :].clone()
        grad_shifted_all = grad_output[:, c*2//3:c, :, :].clone()

        for i in range(ctx.k * 2):
            W_mat_t = ctx.ind[i].permute(0, 2, 1).contiguous()
            grad = grad_shifted_all.view(1, c//3, -1).permute(0, 2, 1)
            grad_shifted_weighted = torch.bmm(W_mat_t, grad)
            grad_shifted_weighted = grad_shifted_weighted.permute(0, 2, 1).contiguous().view(1, c//3, 32, 32)
            grad_latter_all = torch.add(grad_latter_all, grad_shifted_weighted)

        return grad_former_all, grad_latter_all, None, None, None, None, None


def MyUnfold(input, patch_size, stride):
    patches = input.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    i_0, i_1, i_2, i_3, i_4, i_5 = patches.size()
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(i_0, i_2 * i_3 * i_4 * i_5, i_1)
    return patches
