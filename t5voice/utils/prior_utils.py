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
from einops import rearrange
from torch.special import gammaln
import numpy as np


def logbeta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logcombinations(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logbetabinom(n, a, b, x):
    return logcombinations(n, x) + logbeta(x + a, n - x + b) - logbeta(a, b)


def get_beta_binomial_prior(encoder_length: int, decoder_length: int, scaling_factor: float = 1.0) -> np.array:
    x = rearrange(torch.arange(0, encoder_length), "b -> 1 b")
    y = rearrange(torch.arange(1, decoder_length + 1), "b -> b 1")
    a = scaling_factor * y
    b = scaling_factor * (decoder_length + 1 - y)
    n = torch.FloatTensor([encoder_length - 1])
    beta_binomial_prior = logbetabinom(n, a, b, x).exp()

    return beta_binomial_prior