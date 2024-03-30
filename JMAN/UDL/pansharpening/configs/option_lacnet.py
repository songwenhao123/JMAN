import argparse
import os

from UDL.AutoDL import TaskDispatcher


class parser_args(TaskDispatcher, name='LACNET'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()
        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()


        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'./results',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=0.0003, type=float)  # 1e-4 2e-4 8
        parser.add_argument('--lr_scheduler', default=False, type=bool)
        parser.add_argument('--samples_per_gpu', default=16, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--epochs', default=500, type=int)
        parser.add_argument('--workers_per_gpu', default=1, type=int)
        parser.add_argument('--resume_from',
                            default='None',
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='panformer', type=str,
                            choices=['PanNet','GPPNN','LACNET', 'DiCNN', 'PNN', 'FusionNet'])
        parser.add_argument('--act_variant', default='panformer',type=str)
        parser.add_argument('--train_set', default='qb',  choices=['wv3', 'gf2', 'qb'], type=str)
        parser.add_argument('--val_set', default='valid_qb', type=str,
                            choices=['valid_wv3', 'valid_gf2', 'valid_qb',
                                     'test_wv3', 'test_gf2', 'test_qb'])
        parser.add_argument('--eval', action='store_true', 
                            help="performing evalution for patch2entire")

        args, unknown = parser.parse_known_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = "Test"
        cfg.save_fmt = 'mat'
        cfg.img_range = 2047.0
        args.dataset = {'val': args.val_set}
        if not args.eval:
            args.dataset['train'] = args.train_set


        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)
        cfg.workflow = [('train', 50),('val', 1)]
        # cfg.workflow = [('val', 1)]  # only val workflow means perform test.
        # cfg.workflow = [('train', 50)]
        # cfg.dataset = {'train': 'wv3', 'val': 'valid_wv3'}
        self.merge_from_dict(cfg)

