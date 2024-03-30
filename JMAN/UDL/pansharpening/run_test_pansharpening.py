# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import argparse
import sys
sys.path.append("../..")
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main
from models import AVAILABLE_MODELS



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument(
        "--arch",
        default="LANet",
        choices=AVAILABLE_MODELS,
        type=str,
    )
    parser.add_argument("--resume_from", default="'/home/qilei/Experiments/PanCollection/01-DL-toolbox/UDL/pansharpening/results/dbck/LANet/wv3/model_best_400.pth'", type=str)
    
    args, _ = parser.parse_known_args()

    cfg = TaskDispatcher.new(task="pansharpening", mode="entrypoint", arch=args.arch)
    cfg.workflow = [("val", 1)]
    cfg.resume_from = args.resume_from
    print(TaskDispatcher._task.keys())
    print(
        "This is the correct configration: ######################################################################################################################"
    )
    for key, val in cfg.items():
        print(f"{key}: {val}")
    print(
        "This is the end of correct configration: ###########################################################################################################################"
    )
    main(cfg)
