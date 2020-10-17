import numpy as np
import torch
import torch.nn as nn
import functools
from losses import *
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
from attention import MyAttention


class LBPGenerator(nn.Module):
    def __init__(self):
        super(LBPGenerator, self).__init__()
        ngf = 64
        use = False
        self.dn1 = nn.Sequential(
            spectral_norm(nn.Conv2d(2, ngf * 1, kernel_size=4, stride=2, padding=1), use),
        )
        self.dn2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 1, ngf * 2, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 2),
        )
        self.dn3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 4),
        )
        self.dn4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.bottle = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up7 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up6 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up5 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up4 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 4),
        )
        self.up3 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 2),
        )
        self.up2 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 2 * 2, ngf * 1, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 1),
        )
        self.up1 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 1 * 2, 1, kernel_size=4, stride=2, padding=1), use),
            nn.Tanh(),
        )

    def forward(self, lbp, mask):
        dn1 = self.dn1(torch.cat([lbp, 1 - mask], 1))
        dn2 = self.dn2(dn1)
        dn3 = self.dn3(dn2)
        dn4 = self.dn4(dn3)
        dn5 = self.dn5(dn4)
        dn6 = self.dn6(dn5)
        dn7 = self.dn7(dn6)
        bottle = self.bottle(dn7)
        up7 = self.up7(torch.cat([bottle, dn7], 1))
        up6 = self.up6(torch.cat([up7, dn6], 1))
        up5 = self.up5(torch.cat([up6, dn5], 1))
        up4 = self.up4(torch.cat([up5, dn4], 1))
        up3 = self.up3(torch.cat([up4, dn3], 1))
        up2 = self.up2(torch.cat([up3, dn2], 1))
        plbp = self.up1(torch.cat([up2, dn1], 1)) + lbp
        plbp = plbp * mask + lbp * (1 - mask)
        return plbp, [dn1, dn2, dn3, dn4, dn5, dn6, dn7, bottle, up7, up6, up5, up4, up3, up2]


class ImageGenerator(nn.Module):
    def __init__(self, opt):
        super(ImageGenerator, self).__init__()
        use = False
        ngf = 64
        self.dn11 = nn.Sequential(
            spectral_norm(nn.Conv2d(5, ngf * 1, kernel_size=4, stride=2, padding=1), use),
        )
        self.dn21 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 1, ngf * 2, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 2),
        )
        self.dn31 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 4),
        )
        self.dn41 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn51 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn61 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn71 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.bottle1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up71 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up61 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up51 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up41 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 4),
        )
        self.up31 = nn.Sequential(
            nn.ReLU(True),
        )
        self.up311 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf * 4 * 3, ngf * 2, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 2),
        )
        self.up21 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 2 * 2, ngf * 1, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 1),
        )
        self.up11 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 1 * 2, 3, kernel_size=4, stride=2, padding=1), use),
            nn.Tanh(),
        )
        self.myattention = MyAttention(opt)

    def forward(self, x, lbp, mask, second=False):
        dn11 = self.dn11(torch.cat([x, lbp, 1 - mask], 1))
        dn21 = self.dn21(dn11)
        dn31 = self.dn31(dn21)
        dn41 = self.dn41(dn31)
        dn51 = self.dn51(dn41)
        dn61 = self.dn61(dn51)
        dn71 = self.dn71(dn61)
        bottle1 = self.bottle1(dn71)
        up71 = self.up71(torch.cat([bottle1, dn71], 1))
        up61 = self.up61(torch.cat([up71, dn61], 1))
        up51 = self.up51(torch.cat([up61, dn51], 1))
        up41 = self.up41(torch.cat([up51, dn41], 1))
        up31 = self.up31(torch.cat([up41, dn31], 1))
        up31 = self.myattention(up31[:, :256, :, :], up31[:, 256:, :, :], mask)
        up31 = self.up311(up31)
        up21 = self.up21(torch.cat([up31, dn21], 1))
        output = self.up11(torch.cat([up21, dn11], 1)) + x
        output = output * mask + x * (1 - mask)
        return output, [dn11, dn21, dn31, dn41, dn51, dn61, dn71, bottle1, up71, up61, up51, up41, up31, up21]


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_spectral_norm),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_spectral_norm),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_spectral_norm)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = functools.partial(SwitchNorm2d)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, device=''):
    assert(torch.cuda.is_available())
    net.to(device)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(opt):
    netG = ImageGenerator(opt)
    return init_net(netG, 'normal', 0.02, device=opt.device)


def define_LBP(opt):
    netG = LBPGenerator()
    return init_net(netG, device=opt.device)


def define_D(input_nc, ndf, device=''):
    norm_layer = get_norm_layer(norm_type='instance')
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=True, use_spectral_norm=1)
    return init_net(netD, 'normal', 0.02, device)

