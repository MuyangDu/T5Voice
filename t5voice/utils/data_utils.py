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

from .prior_utils import (
    get_beta_binomial_prior
)
from .common_utils import (
    generate_attention_mask
)
from omegaconf import ListConfig
from hydra.utils import to_absolute_path
from tqdm import tqdm
import numpy as np
import torch
import json


class T5VoiceDataset(torch.utils.data.Dataset):

    def __init__(self, filelist_path, tokenizer, min_allowed_duration, max_allowed_duration, logger,
                 encoder_pad_id=0, decoder_pad_id=0, decoder_bos_id=0, 
                 decoder_eos_id=1, label_ignore_id=-100, num_decoder_special_tokens=2,
                 cross_attention_prior_type="beta_binomial", sort_metadata=False):
        self.tokenizer = tokenizer
        self.min_allowed_duration = min_allowed_duration
        self.max_allowed_duration = max_allowed_duration
        self.logger = logger
        self.encoder_pad_id = encoder_pad_id
        self.decoder_pad_id = decoder_pad_id
        self.decoder_bos_id = decoder_bos_id
        self.decoder_eos_id = decoder_eos_id
        self.label_ignore_id = label_ignore_id
        self.num_decoder_special_tokens = num_decoder_special_tokens
        self.cross_attention_prior_type = cross_attention_prior_type
        metadata = self.load_filelist(filelist_path)
        self.metadata = self.filter(metadata)
        if sort_metadata:
            self.sort_metadata_by_duration()
        self.logger.log_message(f"{len(self.metadata)} items left after filtering.")
        self.tokenized_cache = {}
        # self.generate_tokenized_cache()

    def sort_metadata_by_duration(self):
        self.metadata.sort(key=lambda x: x['duration'], reverse=True)
        
    def generate_tokenized_cache(self):
        self.logger.log_message("Generating tokenized cache.")
        for item in tqdm(self.metadata):
            text = item["text"]
            self.tokenized_cache[text] = self.tokenize(text)
        
    def load_filelist(self, path):
        metadata = []
        if isinstance (path, ListConfig):
            for p in path:
                with open(p, 'r', encoding='utf-8') as file:
                    for line in file:
                        metadata.append(json.loads(line.strip()))
        else:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    metadata.append(json.loads(line.strip()))
        return metadata
    
    def filter(self, metadata):
        filtered_metadata = []
        for item in metadata:
            if item["duration"] <= self.max_allowed_duration and item["duration"] >= self.min_allowed_duration:
                filtered_metadata.append(item)
        return filtered_metadata
    
    def tokenize(self, text):
        input_ids, _ = self.tokenizer.encode(text)
        input_ids = torch.LongTensor(input_ids)
        return input_ids
    
    def load_code(self, code_path):
        code = np.load(to_absolute_path(code_path))
        code = torch.LongTensor(code)
        return code
    
    def get_cross_attention_prior(self, encoder_length, decoder_length, type="beta_binomial"):
        if type == "beta_binomial":
            prior = get_beta_binomial_prior(
                encoder_length, decoder_length, scaling_factor=1)
        else:
            raise NotImplementedError(f"The '{type}' prior is not implemented.")
        return prior

    def get_item(self, item):
        code_path = item["code"]
        code = self.load_code(code_path)
        text = item["text"]
        if text not in self.tokenized_cache:
            self.tokenized_cache[text] = self.tokenize(text)
        input_ids = self.tokenized_cache[text]
        decoder_input_ids = code.new_zeros((code.shape[0], code.shape[1] + 1))
        decoder_input_ids[:, 1:] = code + self.num_decoder_special_tokens
        decoder_input_ids[:, 0] = self.decoder_bos_id
        labels = code.new_zeros((code.shape[0], code.shape[1] + 1))
        labels[:, :-1] = code + self.num_decoder_special_tokens
        labels[:, -1] = self.decoder_eos_id
        encoder_input_length = input_ids.shape[0]
        decoder_input_length = decoder_input_ids.shape[1]
        cross_attention_prior = self.get_cross_attention_prior(encoder_input_length, decoder_input_length, type=self.cross_attention_prior_type)

        return input_ids, decoder_input_ids, labels, cross_attention_prior, encoder_input_length, decoder_input_length

    def __getitem__(self, index):
        return self.get_item(self.metadata[index])

    def __len__(self):
        return len(self.metadata)


class T5VoiceCollator():

    def __init__(self, encoder_pad_id=0, decoder_pad_id=0, label_ignore_id=-100):
        self.encoder_pad_id = encoder_pad_id
        self.decoder_pad_id = decoder_pad_id
        self.label_ignore_id = label_ignore_id

    def __call__(self, batch):
        batch_size = len(batch)
        encoder_input_lengths = torch.LongTensor([item[4] for item in batch]).unsqueeze(1)
        decoder_input_lengths = torch.LongTensor([item[5] for item in batch]).unsqueeze(1)
        max_encoder_input_length = encoder_input_lengths.max()
        max_decoder_input_length = decoder_input_lengths.max()
        
        encoder_attention_mask = generate_attention_mask(
            valid_lengths=encoder_input_lengths, 
            padded_length=max_encoder_input_length)
        
        padded_input_ids = torch.zeros((batch_size, max_encoder_input_length), dtype=torch.long) + self.encoder_pad_id
        padded_decoder_input_ids = torch.zeros((batch_size, batch[0][1].shape[0], max_decoder_input_length), dtype=torch.long) + self.decoder_pad_id
        padded_labels = torch.zeros((batch_size, batch[0][1].shape[0], max_decoder_input_length), dtype=torch.long) + self.label_ignore_id
        padded_cross_attention_prior = torch.zeros((batch_size, max_decoder_input_length, max_encoder_input_length))
        
        for index in range(batch_size):
            input_ids = batch[index][0]
            padded_input_ids[index, :input_ids.shape[0]] = input_ids
            decoder_input_ids = batch[index][1]
            padded_decoder_input_ids[index, :, :decoder_input_ids.shape[1]] = decoder_input_ids
            labels = batch[index][2]
            padded_labels[index, :, :labels.shape[1]] = labels
            cross_attention_prior = batch[index][3]
            encoder_input_length = batch[index][4]
            decoder_input_length = batch[index][5]
            padded_cross_attention_prior[index, :decoder_input_length, :encoder_input_length] = cross_attention_prior
        
        padded_decoder_input_ids = padded_decoder_input_ids.permute(0, 2, 1)
        padded_labels = padded_labels.permute(0, 2, 1)

        return padded_input_ids, padded_decoder_input_ids, padded_labels, padded_cross_attention_prior, \
            encoder_input_lengths, decoder_input_lengths, encoder_attention_mask