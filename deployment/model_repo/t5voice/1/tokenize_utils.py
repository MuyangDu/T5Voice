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

import logging

class EnglishIPATokenizer:
    def __init__(self):
        self._pad = "<PAD>"
        self._unk = "<UNK>"
        self._eos = "<EOS>"
        self._bos = "<BOS>"
        self._punctuation = ';:,.!?¡¿—…"«»“” '
        self._letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self._letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        self.tokens = [self._pad] + list(self._punctuation) + list(self._letters) + list(self._letters_ipa) + [self._unk, self._eos, self._bos]
        self.num_tokens = len(self.tokens)
        self.pad_token_id = self.tokens.index(self._pad)
        self.eos_token_id = self.tokens.index(self._eos)
        self.bos_token_id = self.tokens.index(self._bos)
        # self.bos_token_id = self.tokens.index(" ")
        self.unk_token_id = self.tokens.index(self._unk)
        from phonemizer.backend import EspeakBackend
        logger = logging.getLogger('phonemizer')
        logger.setLevel(logging.ERROR)
        self.backend = EspeakBackend('en-us', preserve_punctuation=True, with_stress=True, logger=logger)
        
    def decode(self, token_ids):
        tokens = [self.tokens[token_id] for token_id in token_ids]
        return tokens
        
    def encode(self, text, ref=False):
        tokens = self.backend.phonemize([text], strip=True)[0]
        token_counts = [len(word_tokens) + 1 for word_tokens in tokens.split(" ")]
        token_counts[-1] += 1 # last word is followed by an eos token
        token_ids = [self.bos_token_id]
        token_ids.extend([self.tokens.index(token) if token in self.tokens else self.unk_token_id for token in tokens])
        if ref and tokens[-1] in self._punctuation:
            token_ids[-1] = self.tokens.index(",")
        token_ids.append(self.eos_token_id)
        
        return token_ids, token_counts