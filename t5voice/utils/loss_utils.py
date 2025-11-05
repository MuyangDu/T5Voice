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
import torch.nn as nn
import torch.nn.functional as F


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_scores, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_scores.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_scores = attn_scores.squeeze(1)
        attn_scores = attn_scores.permute(1, 0, 2)

        # Add blank label
        attn_scores = F.pad(
            input=attn_scores,
            pad=(1, 0, 0, 0, 0, 0),
            value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(max_key_len+1, device=attn_scores.device, dtype=torch.long)
        attn_scores.masked_fill_(
            key_inds.view(1, 1, -1) > key_lens.view(1, -1, 1), # key_inds >= key_lens+1
            -1e15)
        attn_logprob = self.log_softmax(attn_scores)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.CTCLoss(
            attn_logprob, target_seqs,
            input_lengths=query_lens, target_lengths=key_lens)
        
        return cost