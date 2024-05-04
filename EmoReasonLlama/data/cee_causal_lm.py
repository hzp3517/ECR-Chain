'''
RECCON数据集，以CausalLM的方式加载Causal Emotion Entrailment任务的输入数据
'''
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import transformers
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from typing import Dict, Optional, Sequence
import json
import os
from transformers.trainer_pt_utils import LabelSmoother
from functools import partial
from toolz.sandbox import unzip
import copy
from tqdm import tqdm

from EmoReasonLlama.data.task_templates.fill_template import *
from EmoReasonLlama.data.conversation import Conversation, get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CEECausalLMDataset(Dataset):
    '''
    Dataset for the causal emotion entailment task by the causal LM type
    '''
    def __init__(self, set_name, tokenizer, raw_data_path, conv_templ_type, task_templ_type, local_rank=0):
        '''
        input:
        - `set_name`: ['train', 'dev', 'test']
        - `tokenizer`: type: transformers.PreTrainedTokenizer
        - `raw_data_path`: the json file of the dataset
        - `conv_templ_type`: specify a conversation template, e.g. ['alpaca', 'vicuna_v1.1']
        - `task_templ_type`: specify task prompt and response template for the target task, e.g. ['CEE_vanilla', 'CEE_answer']
        - `local_rank`: training_args.local_rank
        '''
        super(CEECausalLMDataset, self).__init__()
        self.set_name = set_name
        self.tokenizer = tokenizer
        with open(raw_data_path, 'r', encoding="utf-8") as f:
            self.raw_data_dict = json.load(f)
        self.conv_templ_type = conv_templ_type
        self.task_templ_type = task_templ_type
        self.local_rank = local_rank
        data_dict = self.preprocess(self.raw_data_dict)
        self.input_ids = data_dict["input_ids"]
        self.question_token_len = data_dict["question_token_len"]
        self.attention_mask = data_dict["attention_mask"]
        self.dialog_ids = data_dict["dialog_ids"]
        self.task_targets = data_dict["task_targets"]
        self.num_utts = data_dict["num_utts"]
        if self.set_name == 'train':
            self.labels = data_dict["labels"]
        

    def rank0_print(self, *args):
        if self.local_rank == 0:
            print(*args)

    def preprocess(self, raw_data_dict):
        # for task templates application
        task_templ_file = os.path.join('/data4/hzp/ECR-Chain/EmoReasonLlama/data/task_templates', self.task_templ_type, 'prompt_template.txt')
        assert os.path.exists(task_templ_file)
        with open(task_templ_file, 'r') as f:
            lines = f.readlines()
        prompt_template = ''.join(lines)
        if self.task_templ_type == 'CEE_vanilla':
            fill_conversation_func = fill_conversation_CEE_vanilla
            fill_targetutt_func = fill_targetutt_CEE_vanilla
            fill_response_func = fill_response_CEE_vanilla
        else:
            print(self.task_templ_type)

            raise Exception('Invalid task template name!')

        # for conv templates application
        conv = get_conv_template(self.conv_templ_type) # conv: Conversation类对象
        conv_wo_res = copy.deepcopy(conv) # 用于生成只含问题prompt不含回复内容的输入文本（用于推理阶段）

        # Apply templates for each sample
        conversations = []
        inference_questions = []
        task_targets = []
        num_utts = []
        dia_ids = list(raw_data_dict[self.set_name].keys())

        for dia_id in dia_ids:
            dia_dict = raw_data_dict[self.set_name][dia_id]

            task_targets.append(torch.tensor(dia_dict['pos_cause_utts'], dtype=torch.long))
            num_utts.append(dia_dict['num_utt'])

            # Apply task templates (to prompt)
            task_prompt = prompt_template.replace('<conversation>', fill_conversation_func(dia_dict)).replace('<target_utterance>', fill_targetutt_func(dia_dict))

            # Apply conv_wo_res templates (to prompt) for inference question
            conv_wo_res.messages = []
            conv_wo_res.append_message(conv_wo_res.roles[0], task_prompt)
            conv_wo_res.append_message(conv_wo_res.roles[1], None)
            inference_questions.append(conv_wo_res.get_prompt())

            if self.set_name == 'train':
                # Apply task templates (to response)
                response = fill_response_func(dia_dict)
                # Apply conv templates (to prompt and response) for only single-turn conversation
                conv.messages = []
                conv.append_message(conv.roles[0], task_prompt)
                conv.append_message(conv.roles[1], response)
                conversations.append(conv.get_prompt())

        if self.set_name == 'train': 
            # Tokenize conversations
            input_ids = self.tokenizer(
                conversations,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            targets = input_ids.clone()

            question_token_len = []

            # Mask targets：mask掉非模型生成部分（注意：当前的实现仅支持`padding_side="right"`的情况）
            sep = conv.sep + conv.roles[1] + ": "
            for conversation, target in zip(conversations, targets): # 因为targets是torch.tensor，所以这里直接改target就相当于对targets做了修改
                total_len = int(target.ne(self.tokenizer.pad_token_id).sum()) # ne: not equals, count real length (not padded partion)
                
                # if conv.sep2: # 依照fastchat代码专门对于vicuna_v1.1的处理
                #     conversation = conversation.replace(conv.sep2, '') # vicuna的输入格式中包含用于标记一轮对话结束后的结束符，在我们的任务中直接删掉即可
                
                cur_len = 1 # 跳过<s>
                target[:cur_len] = IGNORE_TOKEN_ID
                if conversation == "":
                    break
                parts = conversation.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                # if conv.sep2: # 依照fastchat代码专门对于vicuna_v1.1的处理
                #     round_len = len(self.tokenizer(rou).input_ids) # tokenizer处理后本身会多出<s>（但不会有</s>），但由于vicuna中sep2已经从rou中去掉，所以抵消掉正好不加不减
                round_len = len(self.tokenizer(conversation).input_ids) - 1 # tokenizer处理后本身会多出<s>，所以想让指针指向下一段对话（多轮对话情况下）的第一个token需要多加一个token
                
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                # 加完sep后的parts[0]应为“xxx[human的话] [这里的空格即sep]Assistant: ”，而tokenzier又会在“xxx”前面自动加入<s>
                # 所以减2表示：减去<s>的长度，以及减去“Assistant:”后面的空格（不mask那个空格）

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                # 留下的每句assistant的话格式如下：“[空格]xxx</s>”，（其中</s>来源于vicuna的sep2）

                question_token_len.append(cur_len + instruction_len)

                cur_len += round_len

                target[cur_len:] = IGNORE_TOKEN_ID # 盖掉后面pad的部分

                if False: # 检查遮蔽的token是否正确
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    self.rank0_print(tokenizer.decode(z))

                if cur_len < self.tokenizer.model_max_length:
                    if cur_len != total_len:
                        target[:] = IGNORE_TOKEN_ID
                        self.rank0_print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" (ignored)"
                        )

            return dict(
                dialog_ids = dia_ids, # hzp add
                input_ids=input_ids,
                labels=targets,
                question_token_len=question_token_len, # hzp add
                task_targets=task_targets, # hzp add
                num_utts=num_utts, # hzp add
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

        else:
            # Tokenize inference_questions
            input_ids = self.tokenizer(
                inference_questions,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            
            question_token_len = [int(i.ne(self.tokenizer.pad_token_id).sum()) for i in input_ids] # ne: not equals, count real length (not padded partion)
            
            return dict(
                dialog_ids = dia_ids, # hzp add
                input_ids=input_ids,
                question_token_len=question_token_len, # hzp add
                task_targets=task_targets, # hzp add
                num_utts=num_utts, # hzp add
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.set_name == 'train':
            return dict(
                dialog_idx=i,
                input_ids=self.input_ids[i],
                labels=self.labels[i], # `labels`专指CausalLM任务的label（即要输出的句子），这个与Trainer.compute_loss()中是关联的。
                question_token_len=self.question_token_len[i],
                attention_mask=self.attention_mask[i],
                task_targets=self.task_targets[i], # 需要手写collate_fn函数，并将其传给`trainer.data_collator`
                num_utts=self.num_utts[i]
            )
        else:
            return dict(
                dialog_idx=i,
                input_ids=self.input_ids[i],
                labels=None,
                question_token_len=self.question_token_len[i],
                attention_mask=self.attention_mask[i],
                task_targets=self.task_targets[i], # 需要手写collate_fn函数，并将其传给`trainer.data_collator`
                num_utts=self.num_utts[i]
            )


    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]):
        """
        Return:
        - dialog_idxs               : (bs,)
        - input_ids                 : (bs, max_len)
        - labels                    : (bs, max_len)
        - attention_mask            : (bs, max_len)
        - task_targets              : (bs, pad_len_#2), i.e. [1, 2, 4, -1, -1] (-1 means pad position)
        - num_utts                  : (bs,) long.
        """
        (dialog_idxs, input_ids, labels, question_token_len, attention_mask, task_targets, num_utts) = map(list, zip(*[d.values() for d in batch]))

        dialog_idxs = torch.tensor(dialog_idxs)

        # 由于已经在preprocess()中做过处理，这里直接把input_ids, labels, attention_mask这三项的batch列表转换成tensor类型即可
        input_ids = torch.stack(input_ids)
        if labels[0] != None:
            labels = torch.stack(labels)
        else: # dev或test set
            labels = None
        question_token_len = torch.tensor(question_token_len)
        attention_mask = torch.stack(attention_mask)

        # for the task targets
        task_targets = pad_sequence(task_targets, batch_first=True, padding_value=-1)
        num_utts = torch.tensor(num_utts)

        return dict(
            dialog_idxs=dialog_idxs,
            input_ids=input_ids,
            labels=labels,
            question_token_len=question_token_len,
            attention_mask = attention_mask,
            task_targets=task_targets,
            num_utts=num_utts
        )