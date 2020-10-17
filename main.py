import cv2
import os
import copy
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from options import MyOptions
from skimage.measure import compare_ssim, compare_psnr
from model import MyModel

opt = MyOptions().parse()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_flist = sorted(os.listdir(self.opt.data_root))
        self.mask_flist = sorted(os.listdir(self.opt.mask_root))

        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        fname = self.img_flist[index]
        I_i = cv2.imread(self.opt.data_root + fname)
        I_g = copy.deepcopy(I_i)
        L_i = self.load_lbp(copy.deepcopy(I_i))
        L_g = copy.deepcopy(L_i)

        I_i = self.transform(I_i)
        I_g = self.transform(I_g)
        L_i = self.transform(L_i)
        L_i = L_i[0, :, :].view(1, 256, 256)
        L_g = self.transform(L_g)
        L_g = L_g[0, :, :].view(1, 256, 256)

        mask = Image.open(self.opt.mask_root + self.mask_flist[index])
        mask = transforms.ToTensor()(mask)
        return {'I_i': I_i, 'I_g': I_g, 'M': mask, 'L_i': L_i, 'L_g': L_g, 'fname': fname}

    def __len__(self):
        return len(self.img_flist)

    def get_pixel(self, img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def lbp_calculated_pixel(self, img, x, y):
        '''
         64 | 128 |   1
        ----------------
         32 |   0 |   2
        ----------------
         16 |   8 |   4
        '''
        center = img[x][y]
        val_ar = []
        val_ar.append(self.get_pixel(img, center, x - 1, y + 1))  # top_right
        val_ar.append(self.get_pixel(img, center, x, y + 1))  # right
        val_ar.append(self.get_pixel(img, center, x + 1, y + 1))  # bottom_right
        val_ar.append(self.get_pixel(img, center, x + 1, y))  # bottom
        val_ar.append(self.get_pixel(img, center, x + 1, y - 1))  # bottom_left
        val_ar.append(self.get_pixel(img, center, x, y - 1))  # left
        val_ar.append(self.get_pixel(img, center, x - 1, y - 1))  # top_left
        val_ar.append(self.get_pixel(img, center, x - 1, y))  # top

        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val

    def load_lbp(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((256, 256, 3), np.uint8)
        for i in range(0, 256):
            for j in range(0, 256):
                img_lbp[i, j, :] = self.lbp_calculated_pixel(img_gray, i, j)
        return img_lbp


class MyDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = MyDataset(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data


def postprocess(img):
    img = img.detach().to('cpu')
    img = img * 127.5 + 127.5
    img = img.permute(0, 2, 3, 1)
    return img.int()


def metrics(real, fake):
    real = postprocess(real)
    fake = postprocess(fake)
    m = (torch.sum(torch.abs(real.float() - fake.float())) / torch.sum(real.float())).float().item()

    a = real.numpy()
    b = fake.numpy()
    ssim = []
    psnr = []
    for i in range(len(a)):
        ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, multichannel=True))
        psnr.append(compare_psnr(a[i], b[i], data_range=255))
    return np.mean(ssim), np.mean(psnr), m


def train():
    opt.device = 'cuda:0'

    opt.data_root = 'demo/input/'   # The location of your training data
    opt.mask_root = 'demo/mask/'    # The location of your training data mask
    train_set = MyDataLoader(opt)

    opt.data_root = 'demo/input/'   # The location of your validation data
    opt.mask_root = 'demo/mask/'    # The location of your validation data mask
    val_set = MyDataLoader(opt)

    model = MyModel()
    model.initialize(opt)

    print('Train/Val with %d/%d' % (len(train_set), len(val_set)))
    for epoch in range(1, 1000):
        print('Epoch: %d' % epoch)
        epoch_iter = 0
        losses_G, ssim, psnr, mae = [], [], [], []
        for i, data in enumerate(train_set):
            epoch_iter += opt.batchSize
            model.set_input(data)
            I_g, I_o, loss_G = model.optimize_parameters()
            s, p, m = metrics(I_g, I_o)
            ssim.append(s)
            psnr.append(p)
            mae.append(m)
            losses_G.append(loss_G.detach().item())
            print('Tra (%d/%d) G:%5.4f, S:%4.4f, P:%4.2f, M:%4.4f' %
                  (epoch_iter, len(train_set), np.mean(losses_G), np.mean(ssim), np.mean(psnr), np.mean(mae)), end='\r')
            if epoch_iter == len(train_set):
                val_ssim, val_psnr, val_mae, val_losses_G = [], [], [], []
                with torch.no_grad():
                    for i, data in enumerate(val_set):
                        fname = data['fname'][0]
                        model.set_input(data)
                        I_g, I_o, val_loss_G = model.optimize_parameters(val=True)
                        val_s, val_p, val_m = metrics(I_g, I_o)
                        val_ssim.append(val_s)
                        val_psnr.append(val_p)
                        val_mae.append(val_m)
                        val_losses_G.append(val_loss_G.item())
                        if i+1 <= 200:
                            cv2.imwrite('./demo/output/' + fname[:-4] + '.png', postprocess(I_o).numpy()[0])
                    print('Val (%d/%d) G:%5.4f, S:%4.4f, P:%4.2f, M:%4.4f' %
                          (epoch_iter, len(train_set), np.mean(val_losses_G), np.mean(val_ssim), np.mean(val_psnr), np.mean(val_mae)))
                losses_G, ssim, psnr, mae = [], [], [], []
        model.save_networks('Model_weights')


def test():
    opt.device = 'cuda:0'
    opt.data_root = 'demo/input/'   # The location of your testing data
    opt.mask_root = 'demo/mask/'    # The location of your testing data mask
    testset = MyDataLoader(opt)
    print('Test with %d' % (len(testset)))

    model = MyModel()
    model.initialize(opt)
    model.load_networks('places_irregular')     # For irregular mask inpainting
    # model.load_networks('celebahq_center')    # For centering mask inpainting, i.e., 120*120 hole in 256*256 input

    val_ssim, val_psnr, val_mae, val_losses_G = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(testset):
            fname = data['fname'][0]
            model.set_input(data)
            I_g, I_o, val_loss_G = model.optimize_parameters(val=True)
            val_s, val_p, val_m = metrics(I_g, I_o)
            val_ssim.append(val_s)
            val_psnr.append(val_p)
            val_mae.append(val_m)
            val_losses_G.append(val_loss_G.detach().item())
            cv2.imwrite('demo/output/' + fname[:-4] + '.png', postprocess(I_o).numpy()[0])
            print('Val (%d/%d) G:%5.4f, S:%4.4f, P:%4.2f, M:%4.4f' % (
                i + 1, len(testset), np.mean(val_losses_G), np.mean(val_ssim), np.mean(val_psnr), np.mean(val_mae)), end='\r')
        print('Val G:%5.4f, S:%4.4f, P:%4.2f, M:%4.4f' %
              (np.mean(val_losses_G), np.mean(val_ssim), np.mean(val_psnr), np.mean(val_mae)))


if __name__ == '__main__':
    if opt.type == 'train':
        train()
    elif opt.type == 'test':
        test()
