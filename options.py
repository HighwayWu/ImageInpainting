import argparse
import os
import util


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [aligned | aligned_resized | single]')
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./weights', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='[instance|batch|switchable] normalization')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')

        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--mask_type', type=str, default='center',
                            help='the type of mask you want to apply, \'center\' or \'random\'')
        parser.add_argument('--mask_sub_type', type=str, default='rect',
                            help='the type of mask you want to apply, \'rect \' or \'fractal \' or \'island \'')
        parser.add_argument('--lambda_A', type=int, default=100, help='weight on L1 term in objective')
        parser.add_argument('--overlap', type=int, default=4, help='the overlap for center mask')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--skip', type=int, default=0, help='Define whether the guidance layer is skipped. Useful when using multiGPUs.')
        parser.add_argument('--fuse', type=int, default=0, help='Fuse may encourage large patches shifting when using \'patch_soft_shift\'')
        parser.add_argument('--gan_type', type=str, default='vanilla', help='wgan_gp, '
                                                                            'lsgan, '
                                                                            'vanilla, '
                                                                            're_s_gan (Relativistic Standard GAN), '
                                                                            're_avg_gan')
        parser.add_argument('--gan_weight', type=float, default=0.2, help='the weight of gan loss')
        parser.add_argument('--style_weight', type=float, default=10.0, help='the weight of style loss')
        parser.add_argument('--content_weight', type=float, default=1.0, help='the weight of content loss')
        parser.add_argument('--discounting', type=int, default=1, help='the loss type of mask part, whether using discounting l1 loss or normal l1')
        parser.add_argument('--use_spectral_norm_D', type=int, default=1, help='whether to add spectral norm to D, it helps improve results')
        parser.add_argument('--use_spectral_norm_G', type=int, default=0, help='whether to add spectral norm in G. Seems very bad when adding SN to G')

        self.initialized = True
        return parser

    def gather_options(self, options=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        self.parser = parser
        if options == None:
            return parser.parse_args()
        else:
            return parser.parse_args(options)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, options=None):

        opt = self.gather_options(options=options)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)
        self.opt = opt
        return self.opt


class MyOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        return parser
