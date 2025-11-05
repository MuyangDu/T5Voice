# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os

from accelerate.utils import set_seed
from omegaconf import open_dict, ListConfig
from .logging_utils import Logger
from hydra.utils import to_absolute_path


def check_args_and_env(args):
    assert args.optim.batch_size % args.optim.grad_acc == 0

    # Train log must happen before eval log
    assert args.eval.every_steps % args.logging.every_steps == 0

    if args.device == 'gpu':
        assert torch.cuda.is_available(), 'We use GPU to train/eval the model'
    

def opti_flags(args):
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == 'bf16' and args.device == 'gpu':
        args.model.add_config.is_bf16 = True


def update_args_with_env_info(args):
    with open_dict(args):
        slurm_id = os.getenv('SLURM_JOB_ID')

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = 'none'

        args.working_dir = os.getcwd()


def update_paths(args):
    args.model.config_path = to_absolute_path(args.model.config_path)
    
    if isinstance(args.data.train_filelist_path, ListConfig):
        args.data.train_filelist_path = [to_absolute_path(p) for p in args.data.train_filelist_path]
    else:
        args.data.train_filelist_path = to_absolute_path(args.data.train_filelist_path)

    if isinstance(args.data.test_filelist_path, ListConfig):
        args.data.test_filelist_path = [to_absolute_path(p) for p in args.data.test_filelist_path]
    else:
        args.data.test_filelist_path = to_absolute_path(args.data.test_filelist_path)

    if args.model.checkpoint_path != "":
        args.model.checkpoint_path = to_absolute_path(args.model.checkpoint_path)

    if args.model.restore_from != "":
        args.model.restore_from = to_absolute_path(args.model.restore_from)
        
        
def setup_basics(accelerator, args):
    check_args_and_env(args)
    update_args_with_env_info(args)
    update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    logger = Logger(args=args, accelerator=accelerator)

    return logger