'''
整理生成后的rationalization数据

postprocess_of_rationalization.json
- dialog_id
    - 'samples'
        - cause_utt_id: str, 一个数字表示当前关注的cause句的序号
            - 'stimulus': 需要在最后加一下对应的cause utt ids
            - 'appraisal' 
            - 'reaction'
            - 'valid_sample': 该cause句对应的样本是否有效（上述三者不为空）
    - 'valid_dialog': 该对话对应的所有缺失cause utt的样本是否都有了有效补充
'''

import os
import csv
import pandas as pd
from tqdm import tqdm
import json
import copy
import re
import sys
sys.path.append('/data4/hzp/ECR-Chain/chatgpt')
from get_prompt_from_template import *

def generation_extract(raw_generation_path):
    generation_dict = {}
    with open(raw_generation_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            dialog_id = row['dialog_id']
            generation = row['generation']
            generation_dict[dialog_id] = generation
    return generation_dict

def postprocess(generation_dict, save_path, set_name='train'):
    '''
    目前是适配supp的版本
    '''
    info_dict = {}
    dialog_dicts = load_set(set_name)
    sample_id_list = sorted(list(generation_dict.keys()))
    for sample_id in tqdm(sample_id_list):
        dialog_id = '_'.join(sample_id.split('_')[:-1])
        if dialog_id not in info_dict.keys():
            info_dict[dialog_id] = {}
            info_dict[dialog_id]['samples'] = {}
        sample_flag = True
        cause_utt_id = sample_id.split('_cause')[-1]
        info_dict[dialog_id]['samples'][cause_utt_id] = {}
        target_speaker = dialog_dicts[dialog_id]['speaker_list'][-1]
        target_utt_id = str(dialog_dicts[dialog_id]['num_utt'])
        generation = generation_dict[sample_id]

        stimulus_raw = None
        appraisal_raw = None
        reaction_raw = None

        # 得到stimulus
        if 'Stimulus in #{}:'.format(cause_utt_id) in generation:
            tmp_gene = generation.split('Stimulus in #{}:'.format(cause_utt_id), 1)[-1].strip()
            if 'Appraisal of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
                stimulus_raw = tmp_gene.split('Appraisal of {} in #{}:'.format(target_speaker, target_utt_id), 1)[0].strip()
            elif 'Appraisal of' in generation:
                stimulus_raw = tmp_gene.split('Appraisal of', 1)[0].strip()
            stimulus_raw = stimulus_raw.splitlines()[0].strip() # 如果有多行，只取第一行
            if stimulus_raw.startswith('- '): # 去掉句首标记
                stimulus_raw = stimulus_raw.split('- ', 1)[-1]
            elif stimulus_raw.startswith('-'):
                stimulus_raw = stimulus_raw.split('-', 1)[-1]
            if stimulus_raw.startswith('None ') or stimulus_raw.startswith('No '): # 去掉未能成功生成的情况
                stimulus_raw = None
            if stimulus_raw:
                if stimulus_raw[-1] == '.': # 加入cause utt id的标记
                    stimulus_raw = stimulus_raw[:-1] + ' (#{}).'.format(cause_utt_id)
                else:
                    stimulus_raw = stimulus_raw + ' (#{}).'.format(cause_utt_id)
        info_dict[dialog_id]['samples'][cause_utt_id]['stimulus'] = stimulus_raw
        if stimulus_raw == None:
            sample_flag = False
        # print(stimulus_raw)
        # break

        # 得到appraisal
        if 'Appraisal of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
            tmp_gene = generation.split('Appraisal of {} in #{}:'.format(target_speaker, target_utt_id), 1)[-1].strip()
            if 'Reaction of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
                appraisal_raw = tmp_gene.split('Reaction of {} in #{}:'.format(target_speaker, target_utt_id), 1)[0].strip()
            elif 'Reaction of' in generation:
                appraisal_raw = tmp_gene.split('Reaction of', 1)[0].strip()
            appraisal_raw = appraisal_raw.splitlines()[0].strip() # 如果有多行，只取第一行
            if appraisal_raw.startswith('- '): # 去掉句首标记
                appraisal_raw = appraisal_raw.split('- ', 1)[-1]
            elif appraisal_raw.startswith('-'):
                appraisal_raw = appraisal_raw.split('-', 1)[-1]
            if appraisal_raw.startswith('None ') or appraisal_raw.startswith('No '): # 去掉未能成功生成的情况
                appraisal_raw = None
        info_dict[dialog_id]['samples'][cause_utt_id]['appraisal'] = appraisal_raw
        if appraisal_raw == None:
            sample_flag = False
            # print(appraisal_raw)
            # break

        # 得到reaction
        if 'Reaction of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
            reaction_raw = generation.split('Reaction of {} in #{}:'.format(target_speaker, target_utt_id), 1)[-1].strip()
            reaction_raw = reaction_raw.splitlines()[0].strip() # 如果有多行，只取第一行
            if reaction_raw.startswith('- '): # 去掉句首标记
                reaction_raw = reaction_raw.split('- ', 1)[-1]
            elif reaction_raw.startswith('-'):
                reaction_raw = reaction_raw.split('-', 1)[-1]
            if reaction_raw.startswith('None ') or reaction_raw.startswith('No '): # 去掉未能成功生成的情况
                reaction_raw = None
        info_dict[dialog_id]['samples'][cause_utt_id]['reaction'] = reaction_raw
        if reaction_raw == None:
            sample_flag = False

        info_dict[dialog_id]['samples'][cause_utt_id]['valid_sample'] = sample_flag

    for dialog_id in tqdm(info_dict.keys()):
        dialog_flag = True
        for s in info_dict[dialog_id]['samples'].keys():
            if info_dict[dialog_id]['samples'][s]['valid_sample'] == False:
                dialog_flag = False
        info_dict[dialog_id]['valid_dialog'] = dialog_flag

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    # ---- set before run ----
    method_dir = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_supp_fewshot'
    raw_generation_file = os.path.join(method_dir, 'raw_generation_for_rationalization.csv')
    save_path = os.path.join(method_dir, 'postprocess_of_rationalization.json')
    # ------------------------

    generation_dict = generation_extract(raw_generation_file)
    postprocess(generation_dict, save_path)