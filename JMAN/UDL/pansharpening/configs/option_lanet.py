import argparse
from UDL.AutoDL import TaskDispatcher
import os
from UDL.pansharpening.models.LANet.act import MODEL_DCIT

class parser_args(TaskDispatcher, name='LANet'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()
        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()

        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # Logger
        parser.add_argument('--out_dir', metavar='DIR', default='./results', help='path to save model')
        parser.add_argument('--mode', default=argparse.SUPPRESS, help='protective declare, please ignore it')

        # parser.add_argument('--lr', default=3e-4, type=float)
        # parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=1024, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=400, type=int)
        parser.add_argument('--workers_per_gpu', default=1, type=int)
        parser.add_argument('--resume_from',
                            default='None',
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        ##
        parser.add_argument('--arch', '-a', metavar='ARCH', default='LANet', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet', 'LANet'])
        parser.add_argument('--train_set', default='qb',  choices=['wv3', 'gf2', 'qb'], type=str)
        parser.add_argument('--act_variant', default='act', choices=MODEL_DCIT.keys(), type=str)
        parser.add_argument('--num_repeats', default=4, type=int)
        parser.add_argument('--val_set', default='valid_qb', type=str,
                            choices=['valid_wv3', 'valid_gf2', 'valid_qb',
                                     'test_wv3', 'test_gf2', 'test_qb'])
        parser.add_argument('--eval', action='store_true', 
                            help="performing evalution for patch2entire")
        
        parser.add_argument('--patch_size', type=int, default=48, 
                        help='input patch size')
        parser.add_argument('--token_size', type=int, default=3, 
                        help='size of token')
        parser.add_argument('--dropout_rate', type=float, default=0, 
                        help='dropout rate for mlp block')
        parser.add_argument('--expansion_ratio', type=int, default=4, 
                        help='expansion ratio for mlp block')
        parser.add_argument('--n_heads', type=int, default=8,
                        help='number of haeds for multi-head self-attention')
        parser.add_argument('--n_layers', type=int, default=8, 
                        help='number of transformer blocks')
        parser.add_argument('--rgb_range', type=int, default=255, 
                        help='maximum value of RGB')
        parser.add_argument('--lr', type=float, default=1e-4, 
                        help='learning rate')
        parser.add_argument('--decay', type=int, default=50, 
                            help='learning rate decay type')
        parser.add_argument('--gamma', type=float, default=0.5,
                            help='learning rate decay factor for step decay')
        parser.add_argument('--momentum', type=float, default=0.9, 
                            help='SGD momentum')
        parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), 
                            help='ADAM beta')
        parser.add_argument('--epsilon', type=float, default=1e-8,
                            help='ADAM epsilon for numerical stability')
        parser.add_argument('--weight_decay', type=float, default=0, 
                        help='weight decay')
        parser.add_argument('--save_path', type=str, default=None, 
                        help='path to save')
        parser.add_argument('--scale', type=int, default=1, 
                        help='super resolution scale')
        parser.add_argument('--self_ensemble', action='store_true', 
                        help='use self-ensemble method for test')
        parser.add_argument('--crop_batch_size', type=int, default=64, 
                        help='input batch size for testing')
        parser.add_argument('--n_colors', type=int, default=4, 
                        help='number of color channels')
        parser.add_argument('--n_feats', type=int, default=64, 
                        help='number of feature maps')
        parser.add_argument('--n_resgroups', type=int, default=4, 
                            help='number of residual groups')
        parser.add_argument('--n_resblocks', type=int, default=12, 
                            help='number of residual blocks')
        parser.add_argument('--reduction', type=int, default=16,
                            help='number of feature maps reduction')

        ## Fusion block
        parser.add_argument('--n_fusionblocks', type=int, default=4, 
                            help='number of fusion blocks')

        args, _ = parser.parse_known_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = 'Test'

        args.dataset = {'val': args.val_set}
        if not args.eval:
            args.dataset['train'] = args.train_set

        cfg.merge_args2cfg(args)
        cfg.save_fmt = "mat"
        # cfg.workflow = [('train', 50), ('val', 1)]

        cfg.use_tfb = False
        cfg.img_range = 2047.0 #1023.0

        self.merge_from_dict(cfg)
