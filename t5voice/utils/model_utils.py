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
from torch.utils.data import DataLoader
from nemo.collections.tts.models import AudioCodecModel
from omegaconf import open_dict
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
from .t5_model import T5Voice
from .tokenize_utils import (
    EnglishIPATokenizer
)
from .data_utils import (
    T5VoiceDataset,
    T5VoiceCollator
)


def get_model(args, config):
    klass = {
        't5voice': T5Voice
    }[args.model.klass]

    if args.model.checkpoint_path and args.model.checkpoint_path.strip() != "":
        model = klass(config)
        state_dict = load_file(args.model.checkpoint_path.strip())
        model.load_state_dict(state_dict, strict=False)
    elif args.model.random_init:
        model = klass(config)
    else:
        raise ValueError(
            "Provide 'checkpoint_path' or set 'random_init' to True."
        )

    with open_dict(args):
        total_params, non_embedding_params, embedding_params = get_model_params(model)
        args.n_all_param = total_params
        args.n_all_param_wo_emb = non_embedding_params
        args.n_all_param_emb = embedding_params
    
    return model


def get_model_params(model):
    total_params = 0
    non_embedding_params = 0
    embedding_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.nelement()
        
        if 'emb' in name.lower():
            embedding_params += param.nelement()
        else:
            non_embedding_params += param.nelement()
    
    return total_params, non_embedding_params, embedding_params


def get_codec_model(args):
    codec_model = AudioCodecModel.from_pretrained(args.codec_model.klass)
    return codec_model


def get_config(args):
    config = AutoConfig.from_pretrained(
        args.model.config_path,
    )

    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f'config does not have attribute {k}'
            setattr(config, k, v)

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f'config already has attribute {k}'
            setattr(config, k, v)
    
    return config


def get_tokenizer(args):
    tokenizer = {
        'english_ipa_tokenizer': EnglishIPATokenizer,
    }[args.model.tokenizer]
    tokenizer = tokenizer()
    
    return tokenizer


def load_dataset_splits(args, tokenizer, logger):
    dataset_class = {
        't5voice': T5VoiceDataset
    }[args.model.klass]

    train_dataset = dataset_class(
        filelist_path=args.data.train_filelist_path,
        tokenizer=tokenizer,
        min_allowed_duration=0.0,
        max_allowed_duration=args.data.max_allowed_duration,
        logger=logger,
        encoder_pad_id=0,
        decoder_pad_id=0,
        decoder_bos_id=0,
        decoder_eos_id=1,
        label_ignore_id=-100,
        num_decoder_special_tokens=2,
        cross_attention_prior_type=args.model.cross_attention_prior_type,
        sort_metadata=False
    )
    
    test_dataset = dataset_class(
        filelist_path=args.data.test_filelist_path,
        tokenizer=tokenizer,
        min_allowed_duration=args.data.min_allowed_duration_for_test,
        max_allowed_duration=args.data.max_allowed_duration,
        logger=logger,
        encoder_pad_id=0,
        decoder_pad_id=0,
        decoder_bos_id=0,
        decoder_eos_id=1,
        label_ignore_id=-100,
        num_decoder_special_tokens=2,
        cross_attention_prior_type=args.model.cross_attention_prior_type,
        sort_metadata=False
    )

    dataset_splits = {
        'train': train_dataset,
        'test': test_dataset
    }

    return dataset_splits


def get_data_collator(args, config):
    collator_class = {
        't5voice': T5VoiceCollator
    }[args.model.klass]

    data_collator = collator_class(
        encoder_pad_id=0, 
        decoder_pad_id=0, 
        label_ignore_id=-100
    )

    return data_collator


def get_dataloaders(tokenizer, config, args, logger):
    dataset_splits = load_dataset_splits(args, tokenizer, logger)
    data_collator = get_data_collator(args, config)

    dataloaders = {}

    for split in ['train', 'test']:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        shuffle = (split == 'train')

        dataloaders[split] = DataLoader(
            dataset_splits[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    with open_dict(args):
        args.data.train_batches = len(dataloaders['train'])
        args.data.test_batches = len(dataloaders['test'])

        if args.optim.epochs > 0:
            args.optim.total_steps = (len(dataloaders['train']) // args.optim.grad_acc) * args.optim.epochs 

        args.eval.corrected_steps = args.eval.steps

    return dataloaders['train'], dataloaders['test']


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )
    elif args.optim.lr_scheduler == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.optim.base_lr if step else 1e-2 / args.optim.base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.optim.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler