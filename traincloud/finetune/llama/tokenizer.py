# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from transformers import GPT2Tokenizer, AutoTokenizer
from typing import List
import os
import torch

logger = getLogger()


class Tokenizer:
    def __init__(self, model: str, model_path: str):
        # reload tokenizer

        if model == "GPTJ6B_adapter":
            self.sp_model = GPT2Tokenizer.from_pretrained( model_path, add_special_tokens = True, local_files_only=True )
            logger.info(f"Reloaded GPT2 Tokenizer model from {model_path}")
            # BOS / EOS token IDs
            self.n_words: int = self.sp_model.vocab_size
            self.bos_id: int = self.sp_model.bos_token_id # 50256
            self.eos_id: int = self.sp_model.eos_token_id # 50256
            self.pad_id: int = self.sp_model.pad_token_id # x
            self.nl: list = [198]
            self.q_token_id : int = 24361
            logger.info(
                f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
            )
        
        elif model == 'OPT6B_adapter':
            self.sp_model = AutoTokenizer.from_pretrained( model_path, add_special_tokens = True, local_files_only=True , use_fast = False)
            logger.info(f"Reloaded GPT2 Tokenizer model from {model_path}")

            # BOS / EOS token IDs
            self.n_words: int = self.sp_model.vocab_size # 50265
            self.bos_id: int = self.sp_model.bos_token_id # 2
            self.eos_id: int = self.sp_model.eos_token_id # 2
            self.pad_id: int = self.sp_model.pad_token_id # 1
            self.nl: list = [50118]
            self.q_token_id : int = 45641
            self.v_token_id : int = 17967
            self.a_token_id : int = 33683

            # TODO placeholder and Question token for OPT [FIND IT]
            logger.info(
                f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
            )

        else:
            model_path=(f'{model_path}./tokenizer.model')
            # assert os.path.isfile(model_path), model_path
            self.sp_model = SentencePieceProcessor(model_file=model_path)
            logger.info(f"Reloaded SentencePiece model from {model_path}")

            # BOS / EOS token IDs
            self.n_words: int = self.sp_model.vocab_size()
            self.bos_id: int = self.sp_model.bos_id()
            self.eos_id: int = self.sp_model.eos_id()
            self.pad_id: int = self.sp_model.pad_id()
            self.nl: list = [13]
            self.q_token_id : int = 16492

            logger.info(
                f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
            )
            assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
