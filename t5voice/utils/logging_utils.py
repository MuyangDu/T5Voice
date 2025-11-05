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

from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger
from omegaconf import OmegaConf, open_dict
import logging
import transformers
import neptune
import os


class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats


class Logger:
    def __init__(self, args, accelerator):
        self.logger = get_logger('Main')

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')
        tensorboard_log_dir = os.path.join(os.getcwd(), "tensorboard")
        self.logger.info(f'Tensorboard log directory is {tensorboard_log_dir}')

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_info()
            self.writer = SummaryWriter(tensorboard_log_dir)
        else:
            transformers.utils.logging.set_verbosity_error()
        
        self.is_local_main_process = accelerator.is_local_main_process

        self.setup_neptune(args)

    def setup_neptune(self, args):
        if args.logging.neptune:
            neptune_logger = neptune.init_run(
                project=args.logging.neptune_creds.project,
                api_token=args.logging.neptune_creds.api_token,
                tags=[str(item) for item in args.logging.neptune_creds.tags.split(",")],
            )
        else:
            neptune_logger = None

        self.neptune_logger = neptune_logger

        with open_dict(args):
            if neptune_logger is not None:
                args.neptune_id = neptune_logger["sys/id"].fetch()

    def log_args(self, args):
        if self.neptune_logger is not None:
            logging_args = OmegaConf.to_container(args, resolve=True)
            self.neptune_logger['args'] = logging_args

    def log_stats(self, stats, step, args, prefix=''):
        if self.neptune_logger is not None:
            for k, v in stats.items():
                self.neptune_logger[f'{prefix}{k}'].log(v, step=step)

        msg_start = f'[{prefix[:-1]}] Step {step} out of {args.optim.total_steps}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.2e}' if k.lower() == "lr" else f'{k.capitalize()} --> {v:.2f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)
        
    def log_tensorboard(self, tag, data, global_step, type="scalar", **kwargs):
        if self.is_local_main_process:
            if type == "scalar":
                self.writer.add_scalar(tag, data, global_step)
            elif type == "image":
                self.writer.add_image(tag, data, global_step)
            elif type == "figure":
                self.writer.add_figure(tag, data, global_step)
            elif type == "audio":
                sample_rate = kwargs.get("sample_rate", 22050)
                self.writer.add_audio(tag, data, global_step, sample_rate=sample_rate)
            else:
                self.logger.warning(f"Logging {type} to TensorBoard is not implemented.")

    def finish(self):
        if self.neptune_logger is not None:
            self.neptune_logger.stop()