import os
import json
import numpy as np
from tqdm import tqdm
import re
import tiktoken
import copy
import random

# set_name: 'test' / 'selected_test' / 'remained_test'

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


def load_simplify_set(set_name, dialog_ids=None):
    '''
    - set_name: ['train', 'dev', 'test']
    - cause_sample_ids: list(e.g. ['tr_223_13', ...]) or None. if specified, only return keys in the list; otherwise, return the whole set.
    
    return:
    - dict of dict
    '''
    if set_name == 'train':
        simplify_subset_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_simplify_fewshot/simplify_subset.json'
    elif set_name == 'dev':
        pass
    elif 'test' in set_name:
        simplify_subset_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_simplify_fewshot/3shot_test/simplify_subset.json'

    with open(dataset_file, 'r', encoding="utf-8") as f:
        raw_data_dict = json.load(f)
    set_data_dict = raw_data_dict[set_name]

    with open(simplify_subset_file, 'r', encoding="utf-8") as f:
        simplify_subset_dict = json.load(f)

    dialog_dicts = {}

    if dialog_ids:
        for dialog_id in dialog_ids:
            assert dialog_id in set_data_dict.keys(), "Cannot find dialog {} in the {} set!".format(dialog_id, set_name)
            dialog_dicts[dialog_id] = set_data_dict[dialog_id]
            dialog_dicts[dialog_id]['sample_id'] = dialog_id
    else:
        for dialog_id in set_data_dict.keys():
            if dialog_id in simplify_subset_dict.keys():
                dialog_dicts[dialog_id] = set_data_dict[dialog_id]
                dialog_dicts[dialog_id]['sample_id'] = dialog_id

    return dialog_dicts




def fill_conversation(dialog_dict):
    content_list = []
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content_list.append("#{}: {} ({}): \"{}\"".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i]))
    content = '\n'.join(content_list)
    return content

def fill_response(dialog_dict, rationales_dict, eval_type):
    if eval_type == 'chain' or eval_type == 'full':
        num_utt = dialog_dict['num_utt']
        target_idx = num_utt - 1
        sample_id = dialog_dict['sample_id']
        theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
        reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
        if len(rationales_dict[sample_id]['reactions_list']):
            for i, reaction_item in enumerate(rationales_dict[sample_id]['reactions_list']):
                assert i+1 == reaction_item['id']
                reaction += "{}. {}\n".format(reaction_item['id'], reaction_item['content'])
        else:
            reaction += "None.\n"
        appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
        if len(rationales_dict[sample_id]['appraisals_list']):
            for i, appr_item in enumerate(rationales_dict[sample_id]['appraisals_list']):
                assert i+1 == appr_item['id']
                appraisal += "{}. {}\n".format(appr_item['id'], appr_item['content'])
        else:
            appraisal += "None.\n"
        stimulus = 'Stimuli:\n'
        if len(rationales_dict[sample_id]['stimuli_list']):
            for i, sti_item in enumerate(rationales_dict[sample_id]['stimuli_list']):
                assert i+1 == sti_item['id']
                stimulus += "{}. {}\n".format(sti_item['id'], sti_item['content'])
        else:
            stimulus += "None.\n"
        content = theme + reaction + appraisal + stimulus
        return content
    elif eval_type == 'cause':
        num_utt = dialog_dict['num_utt']
        target_idx = num_utt - 1
        sample_id = dialog_dict['sample_id']
        stimulus = ''
        if len(rationales_dict[sample_id]['stimuli_list']):
            for i, sti_item in enumerate(rationales_dict[sample_id]['stimuli_list']):
                assert i+1 == sti_item['id']
                stimulus += "{}. {}\n".format(sti_item['id'], sti_item['content'])
        else:
            stimulus += "None.\n"
        return stimulus
    else:
        return None
    


def fill_other_info(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    sample_id = dialog_dict['sample_id']
    target_speaker = dialog_dict['speaker_list'][target_idx]
    target_utt = str(num_utt)
    target_emotion = dialog_dict['emotion_list'][target_idx]
    return target_speaker, target_utt, target_emotion


def get_template(baseline_chain_path, eval_chain_path, eval_type='full', eval_set='test', simplify_set=False): # 读取template文件，并将train set中的examples填入
    '''
    - baseline_chain_path：作为baseline模型的后处理后的chain json文件地址
    - eval_chain_path: 要与baseline做对比的模型的后处理后的chain json文件地址
    - eval_type: 要评估的类型：'chain' / 'cause' / 'full'
    - eval_set: 要评估的set
    - simplify_set：如果要将修正后的推理链作为标准，将该参数设为True
    '''
    eval_root = '/data4/hzp/ECR-Chain/gpt4_eval'
    assert eval_type in ['chain', 'cause', 'full']
    system_file = os.path.join(eval_root, '{}_eval'.format(eval_type), 'system.txt')
    template_file = os.path.join(eval_root, '{}_eval'.format(eval_type), 'prompt_template.txt')
    sample_assignment_file = os.path.join(eval_root, 'sample_assignment.json')

    if simplify_set:
        eval_dialog_dicts = load_simplify_set(eval_set)
    else:
        eval_dialog_dicts = load_set(eval_set)

    # 读两个chain文件：
    with open(baseline_chain_path, 'r', encoding="utf-8") as f:
        baseline_chain_dict = json.load(f)
    with open(eval_chain_path, 'r', encoding="utf-8") as f:
        eval_chain_dict = json.load(f)

    with open(sample_assignment_file, 'r', encoding="utf-8") as f:
        sample_assignment_dict = json.load(f)
    assert eval_set in sample_assignment_dict.keys()
    assign_dict = sample_assignment_dict[eval_set]

    with open(system_file, 'r') as f:
        lines = f.readlines()
    system_content = ''
    lines_num = len(lines)
    i = 0
    while i < lines_num:
        system_content += lines[i]
        i += 1
    
    with open(template_file, 'r') as f:
        lines = f.readlines()
    prompt_templ = ''
    lines_num = len(lines)
    i = 0
    while i < lines_num:
        prompt_templ += lines[i]
        i += 1

    # 替换模板：
    return_dict = {}
    for sample in sorted(list(eval_dialog_dicts.keys())):
        prompt_content = prompt_templ.replace('<conversation>', fill_conversation(eval_dialog_dicts[sample]))
        if assign_dict[sample] == 0: # 值为0：评测时assistant 1是baseline，assistant 2是eval_model
            prompt_content = prompt_content.replace('<ass_1_answer>\n', fill_response(eval_dialog_dicts[sample], baseline_chain_dict, eval_type))
            prompt_content = prompt_content.replace('<ass_2_answer>\n', fill_response(eval_dialog_dicts[sample], eval_chain_dict, eval_type))
        elif assign_dict[sample] == 1:
            prompt_content = prompt_content.replace('<ass_1_answer>\n', fill_response(eval_dialog_dicts[sample], eval_chain_dict, eval_type))
            prompt_content = prompt_content.replace('<ass_2_answer>\n', fill_response(eval_dialog_dicts[sample], baseline_chain_dict, eval_type))
        else:
            raise Exception('Invalid assignment value!')
        target_speaker, target_utt, target_emotion = fill_other_info(eval_dialog_dicts[sample])
        prompt_content = prompt_content.replace('<target_speaker>', target_speaker)
        prompt_content = prompt_content.replace('<target_utt>', target_utt)
        prompt_content = prompt_content.replace('<target_emotion>', target_emotion)

        return_dict[sample] = prompt_content
    return system_content, return_dict



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

def count_tokens(system, prompt):
    messages = [{"role": "system", "content": system}, 
                {"role": "user", "content": prompt}]
    num_tokens = num_tokens_from_messages(messages)
    return num_tokens

def check_token_lens(system, prompt_dict, max_limit=4096):
    '''
    maximum token limit (4,096 for gpt-3.5-turbo or 8,192 for gpt-4)
    '''
    dialog_ids = list(prompt_dict.keys())
    max_len = -1
    longest_id = None
    over_limit_ids = []
    for id in dialog_ids:
        num_tokens = count_tokens(system, prompt_dict[id])
        if num_tokens > max_len:
            longest_id = id
            max_len = num_tokens
        if num_tokens > max_limit:
            over_limit_ids.append(id)
    print('the max len is {}, dialog id: {}'.format(max_len, longest_id))
    print('over_limit_ids:{}'.format(over_limit_ids))