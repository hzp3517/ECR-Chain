'''
对于ECF数据集test集合的数据，基于`prompt_examples`目录下提供的prompt模板，生成可用于chatgpt输入的prompt
'''

import os
import json
import numpy as np
from tqdm import tqdm
from fill_template import *
import re
import tiktoken
import copy
import random

dataset_file = '/data4/hzp/ECR-Chain/RECCON_all_data.json'

def load_set(set_name, dialog_ids=None):
    '''
    - set_name: ['train', 'dev', 'test']
    - dialog_ids: list(e.g. ['0001', '0002', '0003']) or None. if specified, only return keys in the list; otherwise, return the whole set.
    
    return:
    - dict of dict
    '''
    with open(dataset_file, 'r', encoding="utf-8") as f:
        raw_data_dict = json.load(f)
    set_data_dict = raw_data_dict[set_name]

    dialog_dicts = {}

    if dialog_ids:
        for dialog_id in dialog_ids:
            assert dialog_id in set_data_dict.keys(), "Cannot find dialog {} in the {} set!".format(dialog_id, set_name)
            dialog_dicts[dialog_id] = set_data_dict[dialog_id]
            dialog_dicts[dialog_id]['sample_id'] = dialog_id
    else:
        # dialog_dicts = set_data_dict
        for dialog_id in set_data_dict.keys():
            dialog_dicts[dialog_id] = set_data_dict[dialog_id]
            dialog_dicts[dialog_id]['sample_id'] = dialog_id

    return dialog_dicts


def load_rationalization_set(set_name='train', cause_sample_ids=None):
    '''
    - set_name: ['train', 'dev', 'test']
    - cause_sample_ids: list(e.g. ['tr_223_13_cause7', ...]) or None. if specified, only return keys in the list; otherwise, return the whole set.
    
    return:
    - dict of dict
    '''
    if set_name == 'train':
        postprocess_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot/4shot_train/postprocess.json'
    elif set_name == 'dev':
        pass
    elif set_name == 'test':
        postprocess_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot/4shot_full_results/postprocess.json'

    with open(dataset_file, 'r', encoding="utf-8") as f:
        raw_data_dict = json.load(f)
    set_data_dict = raw_data_dict[set_name]

    with open(postprocess_file, 'r', encoding="utf-8") as f:
        postprocess_dict = json.load(f)

    cause_sample_dicts = {}

    if cause_sample_ids:
        for cause_sample_id in cause_sample_ids:
            dialog_id = cause_sample_id.split('_cause')[0]
            cause_id = int(cause_sample_id.split('_cause')[1])
            assert dialog_id in set_data_dict.keys(), "Cannot find dialog {} in the {} set!".format(dialog_id, set_name)
            cause_sample_name = '{}_cause{}'.format(dialog_id, str(cause_id))
            cause_sample_dicts[cause_sample_name] = copy.copy(set_data_dict[dialog_id])
            cause_sample_dicts[cause_sample_name]['dialog_id'] = dialog_id
            cause_sample_dicts[cause_sample_name]['cause_utt_id'] = cause_id
    else:
        for dialog_id in set_data_dict.keys():
            if postprocess_dict[dialog_id]['complete'] == False:
                supp_cause_list = sorted(list(set(postprocess_dict[dialog_id]['answer_label']) - set(postprocess_dict[dialog_id]['answer_pred'])))
                for supp_cause in supp_cause_list:
                    cause_sample_name = '{}_cause{}'.format(dialog_id, str(supp_cause))
                    cause_sample_dicts[cause_sample_name] = copy.copy(set_data_dict[dialog_id])
                    cause_sample_dicts[cause_sample_name]['dialog_id'] = dialog_id
                    cause_sample_dicts[cause_sample_name]['cause_utt_id'] = supp_cause

    return cause_sample_dicts


def load_neg_set(set_name='train', can_sample_ids=None):
    '''
    - set_name: ['train', 'dev', 'test']
    - cause_sample_ids: list(e.g. ['tr_223_13_can7', ...]) or None. if specified, only return keys in the list; otherwise, return the whole set.
    
    return:
    - dict of dict
    '''
    if set_name == 'train':
        postprocess_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot/4shot_train/postprocess.json'
    elif set_name == 'dev':
        pass
    elif set_name == 'test':
        postprocess_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot/4shot_full_results/postprocess.json'

    with open(dataset_file, 'r', encoding="utf-8") as f:
        raw_data_dict = json.load(f)
    set_data_dict = raw_data_dict[set_name]

    with open(postprocess_file, 'r', encoding="utf-8") as f:
        postprocess_dict = json.load(f)

    can_sample_dicts = {}

    if can_sample_ids:
        for can_sample_id in can_sample_ids:
            dialog_id = can_sample_id.split('_can')[0]
            can_id = int(can_sample_id.split('_can')[1])
            assert dialog_id in set_data_dict.keys(), "Cannot find dialog {} in the {} set!".format(dialog_id, set_name)
            can_sample_name = '{}_can{}'.format(dialog_id, str(can_id))
            can_sample_dicts[can_sample_name] = copy.copy(set_data_dict[dialog_id])
            can_sample_dicts[can_sample_name]['dialog_id'] = dialog_id
            can_sample_dicts[can_sample_name]['can_utt_id'] = can_id
    else:
        for dialog_id in set_data_dict.keys():
            neg_can_list = postprocess_dict[dialog_id]['wrong_pred']
            if len(neg_can_list):
                for neg_can in neg_can_list:
                    can_sample_name = '{}_can{}'.format(dialog_id, str(neg_can))
                    can_sample_dicts[can_sample_name] = copy.copy(set_data_dict[dialog_id])
                    can_sample_dicts[can_sample_name]['dialog_id'] = dialog_id
                    can_sample_dicts[can_sample_name]['can_utt_id'] = neg_can
            # 对每个dialog，除正例和上述已经选出的负例外，再随机选取出一个负例
            all_utt = [i for i in range(1, int(dialog_id.split('_')[-1]) + 1)]
            other_neg_can_list = sorted(list(set(all_utt) - set(postprocess_dict[dialog_id]['answer_label']) - set(neg_can_list)))
            if len(other_neg_can_list):
                neg_can = random.choice(other_neg_can_list)
                can_sample_name = '{}_can{}'.format(dialog_id, str(neg_can))
                can_sample_dicts[can_sample_name] = copy.copy(set_data_dict[dialog_id])
                can_sample_dicts[can_sample_name]['dialog_id'] = dialog_id
                can_sample_dicts[can_sample_name]['can_utt_id'] = neg_can

    return can_sample_dicts


def load_simplify_set(set_name='train', dialog_ids=None):
    '''
    - set_name: ['train', 'dev', 'test']
    - cause_sample_ids: list(e.g. ['tr_223_13', ...]) or None. if specified, only return keys in the list; otherwise, return the whole set.
    
    return:
    - dict of dict
    '''
    if set_name == 'train':
        rationalization_subset_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_supp_fewshot/4shot_train/rationalization_subset.json'
    elif set_name == 'dev':
        pass
    elif set_name == 'test':
        rationalization_subset_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_supp_fewshot/4shot_test/rationalization_subset.json'

    with open(dataset_file, 'r', encoding="utf-8") as f:
        raw_data_dict = json.load(f)
    set_data_dict = raw_data_dict[set_name]

    with open(rationalization_subset_file, 'r', encoding="utf-8") as f:
        rationalization_subset_dict = json.load(f)

    dialog_dicts = {}

    if dialog_ids:
        for dialog_id in dialog_ids:
            cause_ids = set_data_dict[dialog_id]['pos_cause_utts']
            assert dialog_id in set_data_dict.keys(), "Cannot find dialog {} in the {} set!".format(dialog_id, set_name)
            dialog_dicts[dialog_id] = set_data_dict[dialog_id]
            dialog_dicts[dialog_id]['sample_id'] = dialog_id
            dialog_dicts[dialog_id]['cause_utt_ids'] = cause_ids
            dialog_dicts[dialog_id]['reactions_list'] = rationalization_subset_dict[dialog_id]['reactions_list']
            dialog_dicts[dialog_id]['appraisals_list'] = rationalization_subset_dict[dialog_id]['appraisals_list']
            dialog_dicts[dialog_id]['stimuli_list'] = rationalization_subset_dict[dialog_id]['stimuli_list']
    else:
        for dialog_id in rationalization_subset_dict.keys():
            cause_ids = set_data_dict[dialog_id]['pos_cause_utts']
            dialog_dicts[dialog_id] = set_data_dict[dialog_id]
            dialog_dicts[dialog_id]['sample_id'] = dialog_id
            dialog_dicts[dialog_id]['cause_utt_ids'] = cause_ids
            dialog_dicts[dialog_id]['reactions_list'] = rationalization_subset_dict[dialog_id]['reactions_list']
            dialog_dicts[dialog_id]['appraisals_list'] = rationalization_subset_dict[dialog_id]['appraisals_list']
            dialog_dicts[dialog_id]['stimuli_list'] = rationalization_subset_dict[dialog_id]['stimuli_list']

    return dialog_dicts


def get_template(template_name, exam_dia_ids=None, exam_set='train'): # 读取template文件，并将train set中的examples填入
    '''
    - template_name: prompt策略名
    - exam_dia_ids: few-shot的样本dialog id列表。 e.g. ['0001', '0002', '0005']。如果是zero-shot就传入[]或None。
    - exam_set: few-shot样本来源set
    '''
    template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    template_file = os.path.join(template_root, template_name, 'template.txt')
    with open(template_file, 'r') as f:
        lines = f.readlines()
    prompt_content = ''
    demon_part = ''
    lines_num = len(lines)
    i = 0
    while i < lines_num:
        if lines[i] != '$$\n':
            prompt_content += lines[i]
            i += 1
        else:
            prompt_content += '<demonstrations>' # 特殊标记，便于后续替换
            i += 1 # 跳过demon part前面的$$
            while lines[i] != '$$\n':
                demon_part += lines[i]
                i += 1
            i += 1 # 跳过demon part后面的$$
    
    fill_targetutt_func = None
    fill_causalutts_func = None
    fill_causeutt_func = None
    fill_canutt_func = None
    fill_causeutts_func = None
    fill_original_chains_func = None
    if template_name == 'CEE_fewshot':
        fill_conversation_func = fill_conversation_CEE_fewshot
        fill_targetutt_func = fill_targetutt_CEE_fewshot
        fill_results_func = fill_results_CEE_fewshot
    elif template_name == 'CEE_v2_fewshot':
        fill_conversation_func = fill_conversation_CEE_v2_fewshot
        fill_targetutt_func = fill_targetutt_CEE_v2_fewshot
        fill_results_func = fill_results_CEE_v2_fewshot
    
    elif template_name == 'CEE_stimuli_fewshot':
        fill_conversation_func = fill_conversation_CEE_stimuli_fewshot
        fill_targetutt_func = fill_targetutt_CEE_stimuli_fewshot
        fill_results_func = fill_results_CEE_stimuli_fewshot
    elif template_name == 'CEE_backward_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_fewshot
        fill_results_func = fill_results_CEE_backward_fewshot
    elif template_name == 'CEE_backward_v2_fewshot' or template_name == 'CEE_backward_v2_zeroshot':
        fill_conversation_func = fill_conversation_CEE_backward_v2_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v2_fewshot
        fill_results_func = fill_results_CEE_backward_v2_fewshot
    elif template_name == 'CEE_backward_1shot' or template_name == 'CEE_backward_1shot':
        fill_conversation_func = fill_conversation_CEE_backward_1shot
        fill_targetutt_func = fill_targetutt_CEE_backward_1shot
        fill_results_func = fill_results_CEE_backward_1shot
    elif template_name == 'CEE_backward_v3_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v3_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v3_fewshot
        fill_results_func = fill_results_CEE_backward_v3_fewshot
    elif template_name == 'CEE_backward_v4_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v4_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v4_fewshot
        fill_results_func = fill_results_CEE_backward_v4_fewshot
    elif template_name == 'CEE_backward_v5_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v5_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v5_fewshot
        fill_results_func = fill_results_CEE_backward_v5_fewshot
    elif template_name == 'CEE_supp_fewshot':
        fill_conversation_func = fill_conversation_CEE_supp_fewshot
        fill_targetutt_func = fill_targetutt_CEE_supp_fewshot
        fill_results_func = fill_results_CEE_supp_fewshot
        fill_causeutt_func = fill_causeutt_CEE_supp_fewshot
    elif template_name == 'CEE_neg_fewshot':
        fill_conversation_func = fill_conversation_CEE_neg_fewshot
        fill_targetutt_func = fill_targetutt_CEE_neg_fewshot
        fill_results_func = fill_results_CEE_neg_fewshot
        fill_canutt_func = fill_canutt_CEE_neg_fewshot
    elif template_name == 'CEE_forward_fewshot':
        fill_conversation_func = fill_conversation_CEE_forward_fewshot
        fill_targetutt_func = fill_targetutt_CEE_forward_fewshot
        fill_causalutts_func = fill_causalutts_CEE_forward_fewshot
        fill_results_func = fill_results_CEE_forward_fewshot
    elif template_name == 'CEE_simplify_fewshot':
        fill_conversation_func = fill_conversation_CEE_simplify_fewshot
        fill_targetutt_func = fill_targetutt_CEE_simplify_fewshot
        fill_causeutts_func = fill_causeutts_CEE_simplify_fewshot
        fill_original_chains_func = fill_original_chains_CEE_simplify_fewshot
        fill_results_func = fill_results_CEE_simplify_fewshot
    else:
        raise Exception('Invalid template name!')
    
    # 需要替换的：<example[x]_conversation>; <example[x]_target_utterance>; <example[x]_results>
    if exam_dia_ids:
        num_exam = len(exam_dia_ids)
    else:
        num_exam = 0
    if exam_dia_ids and num_exam: # few-shot
        if template_name == 'CEE_supp_fewshot':
            dialog_dicts = load_rationalization_set(exam_set, cause_sample_ids=exam_dia_ids)
        elif template_name == 'CEE_neg_fewshot':
            dialog_dicts = load_neg_set(exam_set, can_sample_ids=exam_dia_ids)
        elif template_name == 'CEE_simplify_fewshot':
            dialog_dicts = load_simplify_set(exam_set, dialog_ids=exam_dia_ids)
        else:
            dialog_dicts = load_set(exam_set, dialog_ids=exam_dia_ids)
        full_demon = ''
        for i in range(num_exam):
            tmp_demon_part = demon_part.replace('[x]', str(i+1))
            tmp_demon_part = tmp_demon_part.replace('<example_conversation>\n', fill_conversation_func(dialog_dicts[exam_dia_ids[i]]))
            if fill_targetutt_func:
                tmp_demon_part = tmp_demon_part.replace('<example_target_utterance>\n', fill_targetutt_func(dialog_dicts[exam_dia_ids[i]]))
            if fill_causeutt_func:
                tmp_demon_part = tmp_demon_part.replace('<example_causal_utterance>\n', fill_causeutt_func(dialog_dicts[exam_dia_ids[i]]))
            if fill_canutt_func:
                tmp_demon_part = tmp_demon_part.replace('<example_candidate_utterance>\n', fill_canutt_func(dialog_dicts[exam_dia_ids[i]]))
            if fill_causeutts_func:
                tmp_demon_part = tmp_demon_part.replace('<example_causal_utterances>\n', fill_causeutts_func(dialog_dicts[exam_dia_ids[i]]))
            if fill_causalutts_func:
                tmp_demon_part = tmp_demon_part.replace('<example_causal_utts>\n', fill_causalutts_func(dialog_dicts[exam_dia_ids[i]]))
            if fill_original_chains_func:
                tmp_demon_part = tmp_demon_part.replace('<original_reasoning_chains>\n', fill_original_chains_func(dialog_dicts[exam_dia_ids[i]]))
            tmp_demon_part = tmp_demon_part.replace('<example_results>\n', fill_results_func(dialog_dicts[exam_dia_ids[i]]))
            full_demon += tmp_demon_part
        prompt_content = prompt_content.replace('<demonstrations>', full_demon)

    return prompt_content


def get_testset_prompt(template_name, test_template, test_set='test', dialog_ids=None):
    '''
    - template_name: prompt策略名
    - test_template: get_template()返回结果
    - dialog_ids: list(e.g. ['0001', '0002', '0003']) or None. if specified, only return keys in the list; otherwise, return the whole set.
    return:
    - {dialog_id: prompt}
    '''
    if template_name == 'CEE_supp_fewshot':
        if dialog_ids:
            testset_dialog_dicts = load_rationalization_set(test_set, dialog_ids)
        else:
            testset_dialog_dicts = load_rationalization_set(test_set)
    elif template_name == 'CEE_neg_fewshot':
        if dialog_ids:
            testset_dialog_dicts = load_neg_set(test_set, dialog_ids)
        else:
            testset_dialog_dicts = load_neg_set(test_set)
    elif template_name == 'CEE_simplify_fewshot':
        if dialog_ids:  
            testset_dialog_dicts = load_simplify_set(test_set, dialog_ids)
        else:
            testset_dialog_dicts = load_simplify_set(test_set)
    else:
        if dialog_ids:
            testset_dialog_dicts = load_set(test_set, dialog_ids)
        else:
            testset_dialog_dicts = load_set(test_set)

    fill_targetutt_func = None
    fill_causalutts_func = None
    fill_causeutt_func = None
    fill_canutt_func = None
    fill_causeutts_func = None
    fill_original_chains_func = None
    if template_name == 'CEE_fewshot':
        fill_conversation_func = fill_conversation_CEE_fewshot
        fill_targetutt_func = fill_targetutt_CEE_fewshot
    elif template_name == 'CEE_v2_fewshot':
        fill_conversation_func = fill_conversation_CEE_v2_fewshot
        fill_targetutt_func = fill_targetutt_CEE_v2_fewshot
    elif template_name == 'CEE_stimuli_fewshot':
        fill_conversation_func = fill_conversation_CEE_stimuli_fewshot
        fill_targetutt_func = fill_targetutt_CEE_stimuli_fewshot
    elif template_name == 'CEE_backward_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_fewshot
    elif template_name == 'CEE_backward_v2_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v2_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v2_fewshot
    elif template_name == 'CEE_backward_1shot':
        fill_conversation_func = fill_conversation_CEE_backward_1shot
        fill_targetutt_func = fill_targetutt_CEE_backward_1shot
    elif template_name == 'CEE_backward_v2_zeroshot':
        fill_conversation_func = fill_conversation_CEE_backward_v2_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v2_fewshot
    elif template_name == 'CEE_backward_v3_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v3_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v3_fewshot
    elif template_name == 'CEE_backward_v4_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v4_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v4_fewshot
    elif template_name == 'CEE_backward_v5_fewshot':
        fill_conversation_func = fill_conversation_CEE_backward_v5_fewshot
        fill_targetutt_func = fill_targetutt_CEE_backward_v5_fewshot
    elif template_name == 'CEE_supp_fewshot':
        fill_conversation_func = fill_conversation_CEE_supp_fewshot
        fill_targetutt_func = fill_targetutt_CEE_supp_fewshot
        fill_causeutt_func = fill_causeutt_CEE_supp_fewshot
    elif template_name == 'CEE_neg_fewshot':
        fill_conversation_func = fill_conversation_CEE_neg_fewshot
        fill_targetutt_func = fill_targetutt_CEE_neg_fewshot
        fill_canutt_func = fill_canutt_CEE_neg_fewshot
    elif template_name == 'CEE_forward_fewshot' or template_name == 'CEE_forward_v2_fewshot':
        fill_conversation_func = fill_conversation_CEE_forward_fewshot
        fill_targetutt_func = fill_targetutt_CEE_forward_fewshot
        fill_causalutts_func = fill_causalutts_CEE_forward_fewshot
    elif template_name == 'CEE_simplify_fewshot':
        fill_conversation_func = fill_conversation_CEE_simplify_fewshot
        fill_targetutt_func = fill_targetutt_CEE_simplify_fewshot
        fill_causeutts_func = fill_causeutts_CEE_simplify_fewshot
        fill_original_chains_func = fill_original_chains_CEE_simplify_fewshot
    else:
        raise Exception('Invalid template name!')

    # 替换模板：<test_conversation>
    return_dict = {}
    for sample in sorted(list(testset_dialog_dicts.keys())):
        prompt = test_template.replace('<test_conversation>\n', fill_conversation_func(testset_dialog_dicts[sample]))
        if fill_targetutt_func:
            prompt = prompt.replace('<test_target_utterance>\n', fill_targetutt_func(testset_dialog_dicts[sample]))
        if fill_causeutt_func:
            prompt = prompt.replace('<test_causal_utterance>\n', fill_causeutt_func(testset_dialog_dicts[sample]))
        if fill_canutt_func:
            prompt = prompt.replace('<test_candidate_utterance>\n', fill_canutt_func(testset_dialog_dicts[sample]))
        if fill_causeutts_func:
            prompt = prompt.replace('<test_causal_utterances>\n', fill_causeutts_func(testset_dialog_dicts[sample]))
        if fill_causalutts_func:
            prompt = prompt.replace('<test_causal_utts>\n', fill_causalutts_func(testset_dialog_dicts[sample]))
        if fill_original_chains_func:
            prompt = prompt.replace('<test_original_reasoning_chains>\n', fill_original_chains_func(testset_dialog_dicts[sample]))
        if template_name == 'CEE_backward_v2_zeroshot':
            target_idx = testset_dialog_dicts[sample]['num_utt'] - 1
            target_speaker = testset_dialog_dicts[sample]['speaker_list'][target_idx]
            target_utt_id = str(target_idx + 1)
            prompt = prompt.replace('<target_speaker>', target_speaker)
            prompt = prompt.replace('<target_utt_id>', target_utt_id)
        return_dict[sample] = prompt
    return return_dict

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """
    Returns the number of tokens used by a list of messages.
    Copy from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def count_tokens(prompt):
    messages = [{"role": "user", "content": prompt}]
    num_tokens = num_tokens_from_messages(messages)
    return num_tokens

def check_token_lens(prompt_dict, max_limit=4096):
    '''
    maximum token limit (4,096 for gpt-3.5-turbo or 8,192 for gpt-4)
    '''
    dialog_ids = list(prompt_dict.keys())
    max_len = -1
    longest_id = None
    over_limit_ids = []
    for id in dialog_ids:
        num_tokens = count_tokens(prompt_dict[id])
        if num_tokens > max_len:
            longest_id = id
            max_len = num_tokens
        if num_tokens > max_limit:
            over_limit_ids.append(id)
    print('the max len is {}, dialog id: {}'.format(max_len, longest_id))
    print('over_limit_ids:{}'.format(over_limit_ids))


if __name__ == '__main__':
    # example_dict = load_set('train', dialog_ids=['0001'])
    # print(example_dict)

    # example_dict = load_set('test')
    # print(example_dict)

    # example_dict = load_rationalization_set()
    # print(example_dict.keys())
    # print(len(example_dict))

    # # ----get one sample prompt-------
    # template_name = 'EE2CE_fewshot'
    # exam_dia_ids = ['0001', '0004', '0013']
    # test_dia_id = '0058'
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt = get_one_test_prompt(template_name, prompt_template, test_dia_id)
    # print(prompt)

    # # ----get prompts for the whole testset-----
    # template_name = 'CEE_backward_v2_fewshot'
    # exam_dia_ids = ['tr_264_6', 'tr_5811_4', 'tr_2553_8', 'tr_3449_3']
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt_dict = get_testset_prompt(template_name, prompt_template)
    # print(prompt_dict['tr_9708_4'])

    # print(count_tokens(prompt_dict['tr_9708_4']))
    # check_token_lens(prompt_dict)

    # # ----get prompts for rationalization-----
    # template_name = 'CEE_supp_fewshot'
    # exam_dia_ids = ['tr_1287_9_cause6', 'tr_3676_5_cause3', 'tr_20_8_cause8', 'tr_2232_4_cause3']
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt_dict = get_testset_prompt(template_name, prompt_template, test_set='train')
    # print(prompt_dict['tr_223_2_cause1'])

    # print(count_tokens(prompt_dict['tr_223_2_cause1']))
    # # check_token_lens(prompt_dict)

    # # ----get prompts for neg-----
    # template_name = 'CEE_neg_fewshot'
    # exam_dia_ids = ['tr_1151_3_can3', 'tr_4156_4_can1', 'tr_2683_6_can5', 'tr_2745_4_can3']
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt_dict = get_testset_prompt(template_name, prompt_template, test_set='train')
    # print(prompt_dict['tr_2714_4_can3'])

    # print(count_tokens(prompt_dict['tr_2714_4_can3']))
    # # check_token_lens(prompt_dict)


    # ----get prompts for simplify-----
    template_name = 'CEE_simplify_fewshot'
    exam_dia_ids = ['tr_1319_4', 'tr_132_2', 'tr_1086_3']
    prompt_template = get_template(template_name, exam_dia_ids)
    prompt_dict = get_testset_prompt(template_name, prompt_template, test_set='train')
    # print(prompt_dict['tr_2797_8'])
    print(prompt_dict['tr_1021_6'])

    print(count_tokens(prompt_dict['tr_2797_8']))
    # check_token_lens(prompt_dict)


    # # ----get prompts for backward_v2_zeroshot-----
    # template_name = 'CEE_backward_v2_zeroshot'
    # prompt_template = get_template(template_name)
    # prompt_dict = get_testset_prompt(template_name, prompt_template)
    # print(prompt_dict['tr_9708_4'])

    # print(count_tokens(prompt_dict['tr_9708_4']))
    # # check_token_lens(prompt_dict)


    # # ----get prompts for backward_1shot-----
    # template_name = 'CEE_backward_1shot'
    # exam_dia_ids = ['tr_264_6']
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt_dict = get_testset_prompt(template_name, prompt_template)
    # print(prompt_dict['tr_9708_4'])

    # print(count_tokens(prompt_dict['tr_9708_4']))
    # # check_token_lens(prompt_dict)

    




