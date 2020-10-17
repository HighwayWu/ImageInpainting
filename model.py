import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import util as util
import networks
import torchvision.transforms as transforms
from collections import OrderedDict


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self, loss):
        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.10f' % lr)
        for scheduler in self.schedulers:
            scheduler.step(loss)

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            # if name in ['D', 'D2']:
            #     continue
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.opt.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if name in ['D', 'D2']:
                continue
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.opt.device))
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()): # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class MyModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.model_names = ['G', 'LBP', 'D', 'D2']

        self.netG = networks.define_G(self.opt)
        self.netLBP = networks.define_LBP(self.opt)
        self.netD = networks.define_D(opt.input_nc, opt.ndf, self.opt.device) # Discriminator for netG
        self.netD2 = networks.define_D(opt.input_nc - 2, opt.ndf, self.opt.device) # Discriminator for netLBP

        self.vgg16_extractor = util.VGG16FeatureExtractor().to(self.opt.device)

        self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.opt.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionL1_mask = networks.Discounted_L1(opt).to(self.opt.device)

        self.criterionL2_style_loss = torch.nn.MSELoss()
        self.criterionL2_perceptual_loss = torch.nn.MSELoss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_LBP = torch.optim.Adam(self.netLBP.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        _, self.rand_t, self.rand_l = util.create_rand_mask(self.opt)

    def set_input(self, input):
        I_i = input['I_i'].to(self.opt.device)
        I_g = input['I_g'].to(self.opt.device)
        self.L_i = input['L_i'].to(self.opt.device)
        self.L_g = input['L_g'].to(self.opt.device)

        self.mask = input['M'].to(self.opt.device).narrow(1,0,1)
        self.mask_tmp = input['M'].to(self.opt.device).byte()
        self.mask_tmp = self.mask_tmp.narrow(1,0,1)

        I_i.narrow(1, 0, 1).masked_fill_(self.mask_tmp, 0.)
        I_i.narrow(1, 1, 1).masked_fill_(self.mask_tmp, 0.)
        I_i.narrow(1, 2, 1).masked_fill_(self.mask_tmp, 0.)
        self.L_i.narrow(1, 0, 1).masked_fill_(self.mask_tmp, 0.)

        self.I_i = I_i
        self.I_g = I_g

    def forward(self):
        self.L_o, self.L_fea = self.netLBP(self.L_i, self.mask)
        _, self.I_FEA = self.netG(self.I_g, self.L_g, self.mask)
        self.I_o, self.I_fea = self.netG(self.I_i, self.L_o, self.mask)

    def backward_G(self, val=False):
        I_o = self.I_o
        I_g = self.I_g
        L_o = self.L_o
        L_g = self.L_g

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect':
            # Using the cropped I_o as the input of D.
            I_o = self.I_o[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            I_g = self.I_g[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            L_o = L_o[:, :, self.rand_t:self.rand_t + self.opt.fineSize // 2 - 2 * self.opt.overlap, self.rand_l:self.rand_l + self.opt.fineSize // 2 - 2 * self.opt.overlap]
            L_g = L_g[:, :, self.rand_t:self.rand_t + self.opt.fineSize // 2 - 2 * self.opt.overlap, self.rand_l:self.rand_l + self.opt.fineSize // 2 - 2 * self.opt.overlap]

        pred_I_o = self.netD(I_o)
        pred_L_o = self.netD2(L_o)

        self.loss_G_GAN = self.criterionGAN(pred_I_o, True) * self.opt.gan_weight
        self.loss_G_GAN += self.criterionGAN(pred_L_o, True) * self.opt.gan_weight

        self.loss_G_L2 = 0
        self.loss_G_L2 += self.criterionL2(self.I_o, self.I_g) * 10
        self.loss_G_L2 += self.criterionL2(self.L_o, self.L_g) * self.opt.lambda_A

        vgg_ft_I_o = self.vgg16_extractor(I_o)
        vgg_ft_I_g = self.vgg16_extractor(I_g)
        self.loss_style = 0
        self.loss_perceptual = 0
        for i in range(3):
            self.loss_style += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_I_o[i]), util.gram_matrix(vgg_ft_I_g[i]))
            self.loss_perceptual += self.criterionL2_perceptual_loss(vgg_ft_I_o[i], vgg_ft_I_g[i])

        self.loss_style *= self.opt.style_weight
        self.loss_perceptual *= self.opt.content_weight

        self.loss_multi = 0
        for i in range(len(self.I_fea)):
            self.loss_multi += self.criterionL2(self.I_fea[i], self.I_FEA[i]) * 0.01

        self.loss_G = self.loss_G_L2 + self.loss_G_GAN + self.loss_style + self.loss_perceptual + self.loss_multi

        if val:
            return
        self.loss_G.backward()

    def backward_D(self):
        I_o = self.I_o
        I_g = self.I_g

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect':
            # Using the cropped I_o as the input of D.
            I_o = self.I_o[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            I_g = self.I_g[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]

        self.pred_I_o = self.netD(I_o.detach())
        self.pred_I_g = self.netD(I_g)

        self.loss_D_I_o = self.criterionGAN(self.pred_I_o, False)
        self.loss_D_I_g = self.criterionGAN (self.pred_I_g, True)
        self.loss_D = (self.loss_D_I_o + self.loss_D_I_g) * 0.5
        self.loss_D.backward()

    def backward_D2(self):
        L_o = self.L_o
        L_g = self.L_g

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect':
            # Using the cropped L_o as the input of D.
            L_o = self.L_o[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            L_g = self.L_g[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]

        self.pred_L_o = self.netD2(L_o.detach())
        self.pred_L_g = self.netD2(L_g)

        self.loss_D_L_o = self.criterionGAN(self.pred_L_o, False)
        self.loss_D_L_g = self.criterionGAN(self.pred_L_g, True)
        self.loss_D = (self.loss_D_L_o + self.loss_D_L_g) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self, val=False):
        self.forward()
        self.I_o = self.I_o * self.mask + self.I_g * (1 - self.mask)
        if val:
            self.backward_G(val)
            return self.I_g, self.I_o, self.loss_G

        self.set_requires_grad(self.netD2, True)
        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()
        self.set_requires_grad(self.netD2, False)

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        return self.I_g, self.I_o, self.loss_G