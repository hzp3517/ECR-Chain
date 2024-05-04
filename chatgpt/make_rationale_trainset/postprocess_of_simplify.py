'''
筛选经过简化后刺激条目中能包含所有正确cause utt id的数据，保存成`simplify_subset.json`文件。

出现无关cause utt id以及缺少cause utt id，都直接把这条数据删掉

simplify_subset.json：
- dialog_id
    - 'theme': 对话主题
    - 'stimuli_list':
        - list of dict_stimuli:
            - 'id'
            - 'content'
            - 'corr_utt_ids'
    - 'appraisals_list':
        - list of dict_appraisals:
            - 'id'
            - 'content'
    - 'reactions_list':
        - list of dict_reactions:
            - 'id'
            - 'content'
    - 'answer_label': gt的pos标签句子
    - 'generation': 链条部分（从theme到最后的answer都包含）
        （注意：其中的'\n'都被替换成了'\\n'，用的时候读入json后需要先替换回来）
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


def remove_stimuli_tags(sentence):
    cleaned_sentence = re.sub(r'\s*\(#\d+(?:, #\d+)*\)', '', sentence)
    return cleaned_sentence
def postprocess(generation_dict, save_path, set_name='train'):
    '''
    目前是适配simplify的版本
    '''
    info_dict = {}

    rationalization_subset_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_supp_fewshot/4shot_train/rationalization_subset.json'
    with open(rationalization_subset_file, 'r', encoding="utf-8") as f:
        rationalization_subset_dict = json.load(f)

    dialog_dicts = load_set(set_name)
    for dialog_id in tqdm(list(rationalization_subset_dict.keys())):
        info_dict[dialog_id] = {}
        label = dialog_dicts[dialog_id]['pos_cause_utts']
        target_speaker = dialog_dicts[dialog_id]['speaker_list'][-1]
        target_utt_id = str(dialog_dicts[dialog_id]['num_utt'])

        # --------------- 一、对generation的整理 ----------------
        generation = generation_dict[dialog_id]

        # 1. 分别把theme, reactions_list, appraisals_list, stimuli_list的内容以及pred_stimuli提取出来，如果某一项的模板不匹配，直接视为这一项的信息为None
        theme = None
        reactions_raw = None
        appraisals_raw = None
        stimuli_raw = None
        reactions_list = []
        appraisals_list = []
        stimuli_list = []
        # 1.1 得到theme
        theme = rationalization_subset_dict[dialog_id]['theme']
        # 1.2.1 得到reactions_raw
        if 'Reactions of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
            tmp_gene = generation.split('Reactions of {} in #{}:'.format(target_speaker, target_utt_id), 1)[-1].strip()
            if 'Appraisals of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
                reactions_raw = tmp_gene.split('Appraisals of {} in #{}:'.format(target_speaker, target_utt_id), 1)[0].strip()
            elif 'Appraisals of' in generation:
                reactions_raw = tmp_gene.split('Appraisals of', 1)[0].strip()
        # 1.2.2 得到reactions_list
        if reactions_raw:
            reactions_lines = reactions_raw.splitlines()
            # pattern = r'^(\d+)\.\s(.+)$'
            pattern = r'^(\d+)\.(.+)$'
            '''
            ^: 匹配行的开头。
            (\d+): 这是一个捕获组，它匹配一个或多个数字（0-9）。\d表示数字字符，+表示匹配一个或多个。
            \.: 匹配一个点号。
            \s: 匹配一个空白字符（空格、制表符、换行等）。
            (.+): 这是另一个捕获组，它匹配一个或多个任意字符（除了换行符）。
            $: 匹配行的结尾。
            '''
            reactions_item_ids = []
            for line in reactions_lines:
                match = re.match(pattern, line)
                if match:
                    item_id = int(match.group(1))
                    assert item_id not in reactions_item_ids
                    content = match.group(2).strip()
                    reactions_item_ids.append(item_id)
                    reactions_list.append({'id': item_id, 'content': content})
        # 1.3.1 得到appraisals_raw
        if 'Appraisals of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
            tmp_gene = generation.split('Appraisals of {} in #{}:'.format(target_speaker, target_utt_id), 1)[-1].strip()
            if 'Stimuli:' in generation:
                appraisals_raw = tmp_gene.split('Stimuli:', 1)[0].strip()
        # 1.3.2 得到appraisals_list
        if appraisals_raw:
            appraisals_lines = appraisals_raw.splitlines()
            # pattern = r'^(\d+)\.\s(.+)$'
            pattern = r'^(\d+)\.(.+)$'
            '''
            ^: 匹配行的开头。
            (\d+): 这是一个捕获组，它匹配一个或多个数字（0-9）。\d表示数字字符，+表示匹配一个或多个。
            \.: 匹配一个点号。
            \s: 匹配一个空白字符（空格、制表符、换行等）。
            (.+): 这是另一个捕获组，它匹配一个或多个任意字符（除了换行符）。
            $: 匹配行的结尾。
            '''
            appraisals_item_ids = []
            for line in appraisals_lines:
                match = re.match(pattern, line)
                if match:
                    item_id = int(match.group(1))
                    assert item_id not in appraisals_item_ids
                    content = match.group(2).strip()
                    appraisals_item_ids.append(item_id)
                    appraisals_list.append({'id': item_id, 'content': content})
        # 1.4.1 得到stimuli_raw
        if 'Stimuli:' in generation:
            stimuli_raw = generation.split('Stimuli:', 1)[-1].strip()
        # 1.4.2 得到stimuli_list
        if stimuli_raw:
            stimuli_lines = stimuli_raw.splitlines()
            # pattern = r'^(\d+)\.\s(.+)$'
            pattern = r'^(\d+)\.(.+)$'
            '''
            ^: 匹配行的开头。
            (\d+): 这是一个捕获组，它匹配一个或多个数字（0-9）。\d表示数字字符，+表示匹配一个或多个。
            \.: 匹配一个点号。
            \s: 匹配一个空白字符（空格、制表符、换行等）。
            (.+): 这是另一个捕获组，它匹配一个或多个任意字符（除了换行符）。
            $: 匹配行的结尾。
            '''
            stimuli_item_ids = []
            for line in stimuli_lines:
                match = re.match(pattern, line)
                if match:
                    item_id = int(match.group(1))
                    assert item_id not in stimuli_item_ids
                    content = match.group(2).strip()
                    stimuli_item_ids.append(item_id)
                    corr_utt_ids = []
                    parenthesis_pattern = r'\((.*?)\)'
                    parenthesis_matches = list(re.findall(parenthesis_pattern, content)) # 先找到字符串中所有被小括号包裹的内容
                    sharp_pattern = r'#(\d+)'
                    for m in parenthesis_matches:
                        sharp_matches = re.findall(sharp_pattern, m) # 再找到所有#号后的数字
                        numbers = [int(match) for match in sharp_matches]
                        corr_utt_ids += numbers
                    corr_utt_ids = sorted(list(set(corr_utt_ids)))
                    text = remove_stimuli_tags(content)
                    stimuli_list.append({'id': item_id, 'content': content, 'text': text, 'corr_utt_ids': corr_utt_ids})
        # 1.5 得到pred_stimuli（即stimuli里面保留的条目用#格式显式指出的causal utt id）
        pred_stimuli = []
        for s in stimuli_list:
            pred_stimuli += s['corr_utt_ids']
        pred_stimuli = sorted(list(set(pred_stimuli)))


        # --------------- 二、错误数据的去除 ----------------
        gt_answer = dialog_dicts[dialog_id]['pos_cause_utts']
        if set(gt_answer) != set(pred_stimuli):
            del info_dict[dialog_id]
            continue

        
        # --------------- 三、用筛选和修改后的数据重新构建链条 ----------------
        postprocess_generation = ''
        if theme:
            postprocess_generation += "Theme:\n{}\n".format(theme)
        else:
            postprocess_generation += "Theme:\n{}\n".format('None.')
        postprocess_generation += 'Reactions of {} in #{}:\n'.format(target_speaker, target_utt_id)
        if len(reactions_list):
            for reaction in reactions_list:
                postprocess_generation += "{}. {}\n".format(str(reaction['id']), reaction['content'])
        else:
            postprocess_generation += "None.\n"
        postprocess_generation += 'Appraisals of {} in #{}:\n'.format(target_speaker, target_utt_id)
        if len(appraisals_list):
            for appraisal in appraisals_list:
                postprocess_generation += "{}. {}\n".format(str(appraisal['id']), appraisal['content'])
        else:
            postprocess_generation += "None.\n"
        postprocess_generation += "Stimuli:\n"
        if len(stimuli_list):
            for stimulus in stimuli_list:
                postprocess_generation += "{}. {}\n".format(str(stimulus['id']), stimulus['content'])
        else:
            postprocess_generation += "None.\n"
        postprocess_generation += "Causal utterances:\n{}".format(str(pred_stimuli))
        # --------------------------------------------------


        # --------------- 四、构建json字典 ----------------
        info_dict[dialog_id]['theme'] = theme if theme else 'None.'
        info_dict[dialog_id]['stimuli_list'] = stimuli_list
        info_dict[dialog_id]['appraisals_list'] = appraisals_list
        info_dict[dialog_id]['reactions_list'] = reactions_list
        info_dict[dialog_id]['answer_label'] = label
        info_dict[dialog_id]['generation'] = postprocess_generation.replace('\n', '\\n') # 在标准的JSON中，字符串内部不允许出现未转义的换行符
        # --------------------------------------------------


    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    # ---- set before run ----
    method_dir = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_simplify_fewshot'
    raw_generation_file = os.path.join(method_dir, '3shot_train', 'raw_generation_for_simplify.csv')
    save_path = os.path.join(method_dir, 'simplify_subset.json')
    # ------------------------

    generation_dict = generation_extract(raw_generation_file)
    postprocess(generation_dict, save_path)

    cot_method_dir = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot'
    cot_postprocess_file = os.path.join(cot_method_dir, '4shot_train', 'postprocess.json')
    with open(cot_postprocess_file, 'r', encoding="utf-8") as f:
        postprocess_dict = json.load(f)
    total_num = len(list(sorted(postprocess_dict.keys())))
    with open(save_path, 'r', encoding="utf-8") as f:
        save_dict = json.load(f)
    complete_num = len(list(save_dict.keys()))
    print('complete num: {}, total num: {}, complete ratio: {}'.format(complete_num, total_num, complete_num*1.0/total_num))