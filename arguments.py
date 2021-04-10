import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--img_activation', default='sigmoid',
                            help="activation on the image features")
        parser.add_argument('--sound_activation', default='sigmoid',
                            help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--binary_mask', default=1, type=int,
                            help="whether to use bianry masks")
        parser.add_argument('--mask_thres', default=0.55, type=float,
                            help="threshold in the case of binary masks")
        parser.add_argument('--loss', default='bce',
                            help="loss function to use")
        parser.add_argument('--init', default='no',
                            help="The initiation method fow parameters")

        parser.add_argument('--num_gpus', default=4, type=int,
                            help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=16, type=int,
                            help='number of data loading workers')

        parser.add_argument('--audLen', default=65280, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")

        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')

        parser.add_argument('--seed', default=0, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./ckpt',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')

        parser.add_argument('--path', default='',
                            help="The path to the first layer of dataset")
        parser.add_argument('--jason_name', default='',
                            help="The name of the jason file at the path")
        parser.add_argument('--type_dir', default='Videos_solo',
                            help="The music type choosen for experiment")
        parser.add_argument('--fea_length', default=16, type=int,
                            help="The length of the clip")
        parser.add_argument('--sample_method', default='uniform',
                            help="uniform or dense sampling of clip frames")
        parser.add_argument('--fixed_interval', action='store_true',
                            help="Fixed interval sampling for uniform")

        # The audio preprocessing
        parser.add_argument('--stft_length', default=1022, type=int,
                            help="The length of stft")
        parser.add_argument('--sr', default=11025, type=int,
                            help="The sampling rate")
        parser.add_argument('--wave_length', default=65280, type=int,
                            help="The length of samped sound")
        parser.add_argument('--mix_num', default=2, type=int,
                            help="The number of mixture")

        # Add the discription for the model.
        parser.add_argument('--discription', default='debug',
                            help="Add your discription")

        # Distributed trainning.
        parser.add_argument('--local_rank', default=0, type=int,
                            help='rank of the process')
        # Motion net.
        parser.add_argument('--motion_path', default='',
                            help='The path of the motion net')

        #
        parser.add_argument('--reuse', default='None',
                            help='The path to reuse the model')
        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='data/train.csv')
        parser.add_argument('--list_val',
                            default='data/val.csv')
        parser.add_argument('--dup_trainset', default=100, type=int,
                            help='duplicate so that one epoch has more iters')

        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_synthesizer',
                            default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_steps',
                            nargs='+', type=int, default=[40, 50, 60],
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')

        self.parser = parser

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        return args
