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
import torch._dynamo
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np


def generate_attention_mask(valid_lengths, padded_length):
    attention_mask = torch.arange(padded_length, device=valid_lengths.device).unsqueeze(0) < valid_lengths
    return attention_mask.long()


@torch.jit.script
def batched_top_k_filtering(scores: torch.FloatTensor, top_k: torch.Tensor, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    B, N, M = scores.shape
    top_k = torch.maximum(top_k, torch.tensor(min_tokens_to_keep, device=top_k.device))
    max_top_k = top_k.max().item()
    top_k_values, _ = torch.topk(scores, max_top_k, dim=-1)
    top_k_thresholds = top_k_values.gather(-1, (top_k - 1).unsqueeze(-1).unsqueeze(-1).expand(B, N, 1)).squeeze(-1)
    mask = scores < top_k_thresholds.unsqueeze(-1)
    scores = scores.masked_fill(mask, filter_value)
    
    return scores


@torch.jit.script
def batched_top_p_filtering(scores: torch.FloatTensor, top_p: torch.Tensor, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    if not ((top_p > 0) & (top_p <= 1)).all():
        raise ValueError("All values in `top_p` must be between 0 and 1.")
    B, N, M = scores.shape
    sorted_logits, sorted_indices = torch.sort(scores, descending=False, dim=-1)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    top_p = top_p.unsqueeze(-1).unsqueeze(-1).expand(B, N, M)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, filter_value)
    
    return scores


def cross_attention_prior_annealing(cross_attention_prior, current_train_step, start_scale_down_step, end_step):
    if current_train_step >= end_step:
        return None
    elif current_train_step > start_scale_down_step and current_train_step < end_step:
        total_annealing_steps = end_step - start_scale_down_step
        curr_annealing_step = current_train_step - start_scale_down_step
        cross_attention_prior = cross_attention_prior + ((1.0 - cross_attention_prior) * curr_annealing_step / total_annealing_steps)
        return cross_attention_prior
    else:
        return cross_attention_prior


def plot_alignment_to_figure(alignment, info=None, xlabel='Decoder timestep', ylabel='Encoder timestep'):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig


def merge_figures(figures):
    N = len(figures)
    fig, axes = plt.subplots(1, N, figsize=(N * 6, 4))
    if N == 1:
        axes = [axes]
    for i, figure in enumerate(figures):
        ax = axes[i]
        canvas = FigureCanvas(figure)
        canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(height), int(width), 4)
        ax.imshow(image)
        ax.axis('off')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.close()
    return fig