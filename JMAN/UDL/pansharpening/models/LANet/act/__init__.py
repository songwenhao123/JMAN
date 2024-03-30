import os
import requests
from os import path as osp

import torch

from .act_net import ACT, LNBase, LNSwap, LNContrastSwap, LWLNBase, LWLNSwap, LWLNContrastSwap, NoInteraction, InterInteraction, IntraInteraction
from .act_module import ACTLitModule

MODEL_DCIT = {
    'act': ACT,
    'ln_base': LNBase,
    'ln_swap': LNSwap,
    'lwln_base': LWLNBase,
    'ln_contrastswap': LNContrastSwap,
    'lwln_swap': LWLNSwap,
    'lwln_contrastswap': LWLNContrastSwap,
    'no_interaction': NoInteraction,
    'inter_interaction': InterInteraction,
    'intra_interaction': IntraInteraction
}

def model_factory(model_name):
    return MODEL_DCIT[model_name]

def create_act_model(args, is_train=False):
    net = model_factory(args.act_variant)(args)

    if is_train:
        return ACTLitModule(net=net, args=args)

    else: # test setting
        if args.release:
            model_path = f'pretrained_weights/act_{args.task}_x{args.scale}.pt'
            if not osp.exists(model_path):
                # download pretrained weight
                os.makedirs(osp.dirname(model_path), exist_ok=True)
                url = f'https://github.com/jinsuyoo/ACT/releases/download/v0.0.0/{osp.basename(model_path)}'
                r = requests.get(url, allow_redirects=True)
                print(f'Downloading pretrained weight: {model_path}')
                open(model_path, 'wb').write(r.content)
            
            net.load_state_dict(torch.load(model_path))

            return ACTLitModule(net=net, args=args)

        elif args.ckpt_path is not None:
            # use pretrained parameter
            assert osp.exists(args.ckpt_path), print(f'checkpoint not exists: {args.ckpt_path}')
            print(f'Loading checkpoint from: {args.ckpt_path}')

            return ACTLitModule.load_from_checkpoint(args.ckpt_path, args=args)
        
        else:
            raise ValueError('Need release option or checkpoint path')