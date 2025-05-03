# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 use_outside_answer=False,
                 max_prompt_length=1024,
                 max_answer_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.use_outside_answer = use_outside_answer 

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                            axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        if self.use_outside_answer:
            answer = row_dict['reward_model']['answer']
            # for outside response, always truncate on the right side
            input_ids_answer, attention_mask_answer = verl_F.tokenize_and_postprocess_data(prompt=answer,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.max_answer_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation='right')


            '''            idx = input_ids[0]  # (bs, prompt_length)
                        # left-padded attention_mask
                        attention_mask = attention_mask[0]
                        position_ids = position_ids[0]
                        response = input_ids_answer[0]
                        seq = torch.cat([idx, response], dim=-1)


                        

                        # used to construct attention_mask
                        eos_token_id = self.tokenizer.eos_token_id
                        batch_size = idx.size(0)

                        response_length = response.size(1)
                        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

                        response_position_ids = position_ids[:, -1:] + delta_position_id
                        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
                        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            '''
            position_ids_answer = compute_position_id_with_mask(attention_mask_answer)
            row_dict['input_ids_answer'] = input_ids_answer[0]
            row_dict['attention_mask_answer'] = attention_mask_answer[0]
            row_dict['position_ids_answer'] = position_ids_answer[0]       

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
