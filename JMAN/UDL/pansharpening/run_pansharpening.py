import argparse
import sys

sys.path.append("../..")
from clearml import Task
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main
from models import AVAILABLE_MODELS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--arch', default='LANet', choices=AVAILABLE_MODELS, type=str)
    parser.add_argument("--clearml_proj", default="Pans@RS", type=str)
    parser.add_argument("--clearml_task", default="debug", type=str)
    parser.add_argument("--start_val", default=50, type=int)

    args, _ = parser.parse_known_args()

    # task = Task.init(project_name=args.clearml_proj, task_name=args.clearml_task)

    cfg = TaskDispatcher.new(mode="entrypoint", task="pansharpening", arch=args.arch)

    cfg.workflow = [("train", args.start_val), ("val", 1)]

    print(TaskDispatcher._task.keys())

    main(cfg)
