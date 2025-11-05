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

from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time
import os

from .utils import (
    setup_basics,
    train,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_codec_model,
    get_dataloaders,
    get_config,
)


@hydra.main(config_path="configs", config_name="t5voice_default", version_base='1.1')
def main(args):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config)
    codec_model = get_codec_model(args)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args, logger)

    logger.log_args(args)
    logger.log_message(f"Total Params: {args.n_all_param} Total Params wo Emb: {args.n_all_param_wo_emb}")
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    with open_dict(args):
        args.current_train_step = 1
        args.last_log = time.time()

    if args.model.restore_from != "":
       accelerator.load_state(args.model.restore_from)
       scheduler_path = os.path.join(args.model.restore_from, "scheduler.bin")
       if os.path.exists(scheduler_path):
           lr_scheduler.load_state_dict(torch.load(scheduler_path))
       logger.log_message(f"Restored from {args.model.restore_from}")

       scheduler_state_dict = lr_scheduler.state_dict()
       with open_dict(args):
           args.current_train_step = scheduler_state_dict["last_epoch"] + 1

    if args.model.compile:
        model = torch.compile(model)

    train(model, train_dataloader, test_dataloader, accelerator, lr_scheduler, optimizer, logger, args, tokenizer, codec_model)

    logger.finish()


if __name__ == "__main__":
    main()