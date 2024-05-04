'''
筛选现有推理链中正确的答案对应的链条，多出来的对应答案不在label范围内的链条直接删掉。保存成`postprocess.json`文件。

postprocess.json：
- dialog_id
    - 'theme': 对话主题
    - 'stimuli_list':
        - list of dict_stimuli:
            - 'id'
            - 'content'
            - 'corr_appraisals_ids'
            - 'corr_utt_ids'
    - 'appraisals_list':
        - list of dict_appraisals:
            - 'id'
            - 'content'
            - 'corr_reactions_ids'
    - 'reactions_list':
        - list of dict_reactions:
            - 'id'
            - 'content'
    - 'answer_label': gt的pos标签句子
    - 'answer_pred': 预测的cause句子（已排除不在标签范围内的句子）
    - 'wrong_pred': 预测中不在标签范围内的句子
    - 'complete': True or False. 如果当前保存的推理链已包含所有正确的cause句，则为True，否则为False
    - 'generation': 经过筛除错误链条后，剩下的链条部分（从theme到最后的answer都包含）
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


def postprocess(generation_dict, save_path, set_name='train'):
    '''
    目前是适配backward_v2 CoT的版本
    '''
    info_dict = {}

    dialog_dicts = load_set(set_name)
    for dialog_id in tqdm(list(dialog_dicts.keys())):
        info_dict[dialog_id] = {}
        assert dialog_id in generation_dict.keys()
        label = dialog_dicts[dialog_id]['pos_cause_utts']
        target_speaker = dialog_dicts[dialog_id]['speaker_list'][-1]
        target_utt_id = str(dialog_dicts[dialog_id]['num_utt'])

        # --------------- 一、对generation的整理 ----------------
        generation = generation_dict[dialog_id]

        # 1. 分别把theme, reactions_list, appraisals_list, stimuli_list的内容以及pred_stimuli, pred_final提取出来，如果某一项的模板不匹配，直接视为这一项的信息为None
        theme = None
        reactions_raw = None
        appraisals_raw = None
        stimuli_raw = None
        reactions_list = []
        appraisals_list = []
        stimuli_list = []
        # 1.1 得到theme
        if 'Theme:' in generation:
            theme = generation.strip().split('Theme:')[-1].strip().split('\n', 1)[0]
            if theme.startswith('Unknown') or theme.startswith('unknown') or theme.startswith('None') or theme.startswith('none'):
                theme = None # 有些数据可能难以提取出主题，有时除了'Unknown'外还会生成一些对于难以提取的解释。
        # 1.2.1 得到reactions_raw
        if 'Reactions of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
            tmp_gene = generation.split('Reactions of {} in #{}:'.format(target_speaker, target_utt_id), 1)[-1].strip()
            if 'Appraisals of {} in #{}:'.format(target_speaker, target_utt_id) in generation:
                reactions_raw = tmp_gene.split('Appraisals of {} in #{}:'.format(target_speaker, target_utt_id), 1)[0].strip()
            elif 'Appraisals of' in generation:
                reactions_raw = tmp_gene.split('Appraisals of', 1)[0].strip()
        # 1.2.2 得到reactions_list
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
        appraisals_lines = appraisals_raw.splitlines()
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
        appraisals_list = []
        for line in appraisals_lines:
            match = re.match(pattern, line)
            if match:
                item_id = int(match.group(1))
                assert item_id not in appraisals_item_ids
                tmp_content = match.group(2)
                if tmp_content.find('For Reaction ') >= 0:
                    tmp_content = tmp_content.strip().split('For Reaction ', 1)[-1] # 1: xxxx. / 1 & 2: xxx.
                    cand_reactions = re.findall(r'\d+', tmp_content.strip().split(':', 1)[0])
                    cand_reactions_list = sorted(list(set([int(num) for num in cand_reactions]))) # [1] / [1, 2]
                    corr_reactions_list = []
                    for cand in cand_reactions_list: #  还需要逐个检查一下id是否真的存在于reaction中
                        if cand in reactions_item_ids:
                            corr_reactions_list.append(cand)
                    if len(corr_reactions_list):
                        content = tmp_content.strip().split(':', 1)[-1].strip()
                        appraisals_item_ids.append(item_id)
                        appraisals_list.append({'id': item_id, 'content': content, 'corr_reactions_ids': corr_reactions_list})
        # 1.4.1 得到stimuli_raw
        if 'Stimuli:' in generation:
            tmp_gene = generation.split('Stimuli:', 1)[-1].strip()
            if 'Causal utterances:' in generation:
                stimuli_raw = tmp_gene.split('Causal utterances:', 1)[0].strip()
        # 1.4.2 得到stimuli_list
        stimuli_lines = stimuli_raw.splitlines()
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
        stimuli_list = []
        for line in stimuli_lines:
            match = re.match(pattern, line)
            if match:
                item_id = int(match.group(1))
                assert item_id not in stimuli_item_ids
                tmp_content = match.group(2)
                if tmp_content.find('For Appraisal ') >= 0:
                    tmp_content = tmp_content.strip().split('For Appraisal ', 1)[-1] # 1: xxxx. / 1 & 2: xxx.
                    cand_appraisals = re.findall(r'\d+', tmp_content.strip().split(':', 1)[0])
                    cand_appraisals_list = sorted(list(set([int(num) for num in cand_appraisals]))) # [1] / [1, 2]
                    corr_appraisals_list = []
                    for cand in cand_appraisals_list: #  还需要逐个检查一下id是否真的存在于appraisals中
                        if cand in appraisals_item_ids:
                            corr_appraisals_list.append(cand)
                    if len(corr_appraisals_list):
                        content = tmp_content.strip().split(':', 1)[-1].strip()
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
                        stimuli_list.append({'id': item_id, 'content': content, 'corr_appraisals_ids': corr_appraisals_list, 'corr_utt_ids': corr_utt_ids})
        # 1.5 得到pred_stimuli（即stimuli里面保留的条目用#格式显式指出的causal utt id）
        pred_stimuli = []
        for s in stimuli_list:
            pred_stimuli += s['corr_utt_ids']
        pred_stimuli = sorted(list(set(pred_stimuli)))
        # 1.6 得到pred_final
        tmp_gene = generation.strip().split('Causal utterances:')[-1] # 找到"Causal utterances:"后面的部分
        query = re.search("\[.*?\]", tmp_gene, re.I|re.M) # 找到满足条件（前后有中括号）的最短匹配 https://blog.csdn.net/weixin_42219542/article/details/109451826
        if query:
            query_str = query.group() # query.group()：得到字符串
        else:
            query_str = ''
        try:
            pred_final = eval(query_str) # eval(): 将字符串形式转成列表形式
        except Exception as ex:
            pred_final = []
        pred_final = sorted(list(set(pred_final))) # 去重操作
        # --------------------------------------------------

        # --------------- 二、错误项的去除 ----------------
        remained_reactions_list = []
        remained_appraisals_list = []
        remained_stimuli_list = []

        # 1. 筛stimuli_list（同时修正句中的#项），并判断该数据的推理链是否关于所有answer完整（得到`complete`变量）
        delete_utt_ids = []
        for stimulus in stimuli_list:
            s_corr_utt_ids = copy.deepcopy(stimulus['corr_utt_ids'])
            s_delete_utt_ids = []
            if len(s_corr_utt_ids) == 0: # 原本就没有对应上任何上下文句子的刺激条目应该被留下
                remained_stimuli_list.append(stimulus)
            else:
                remain = False # 此项stimulus是否要保留的状态位。只要存在一项正确的utt id，就应改为True
                for pred_utt_id in s_corr_utt_ids:
                    if pred_utt_id in label:
                        remain = True
                    else:
                        stimulus['corr_utt_ids'].remove(pred_utt_id)
                        s_delete_utt_ids.append(pred_utt_id)
                        delete_utt_ids.append(pred_utt_id)
                if remain:
                    # 需要删除content中对应的序号部分
                    for del_id in s_delete_utt_ids:
                        stimulus['content'] = stimulus['content'].replace(', #{}'.format(str(del_id)), '')
                        stimulus['content'] = stimulus['content'].replace(',#{}'.format(str(del_id)), '')
                        stimulus['content'] = stimulus['content'].replace('(#{}, '.format(str(del_id)), '(')
                        stimulus['content'] = stimulus['content'].replace('(#{},'.format(str(del_id)), '(')
                        stimulus['content'] = stimulus['content'].replace('#{}'.format(str(del_id)), '')
                        stimulus['content'] = stimulus['content'].replace('()'.format(str(del_id)), '')
                    remained_stimuli_list.append(stimulus)
        complete = True # 该条数据的stimuli覆盖是否完整的状态位。只要存在一项标签中存在而对应的stimuli项不存在的情况，就应改为False
        for label_utt_id in label:
            if label_utt_id not in pred_stimuli:
                complete = False

        # 2. 根据stimuli字典筛appraisal列表
        remained_appraisals_ids = []
        for s in remained_stimuli_list:
            remained_appraisals_ids += s['corr_appraisals_ids']
        remained_appraisals_ids = sorted(list(set(remained_appraisals_ids)))
        for appraisal in appraisals_list:
            if appraisal['id'] in remained_appraisals_ids:
                remained_appraisals_list.append(appraisal)

        # 3. 根据appraisals字典筛reaction列表
        remained_reactions_ids = []
        for a in remained_appraisals_list:
            remained_reactions_ids += a['corr_reactions_ids']
        remained_reactions_ids = sorted(list(set(remained_reactions_ids)))
        for reaction in reactions_list:
            if reaction['id'] in remained_reactions_ids:
                remained_reactions_list.append(reaction)
        # --------------------------------------------------

        # --------------- 三、调整每一项的序号 ----------------
        # 1. 调整reactions的序号（包括appraisals中的'corr_reactions_ids'项）
        for idx, reaction in enumerate(remained_reactions_list):
            if reaction['id'] != idx + 1:
                for appraisal in remained_appraisals_list:
                    while reaction['id'] in appraisal['corr_reactions_ids']:
                        corr_idx = appraisal['corr_reactions_ids'].index(reaction['id'])
                        appraisal['corr_reactions_ids'][corr_idx] = idx + 1
                reaction['id'] = idx + 1
        # 2. 调整appraisals的序号（包括stimuli中的'corr_appraisals_ids'项）
        for idx, appraisal in enumerate(remained_appraisals_list):
            if appraisal['id'] != idx + 1:
                for stimulus in remained_stimuli_list:
                    while appraisal['id'] in stimulus['corr_appraisals_ids']:
                        corr_idx = stimulus['corr_appraisals_ids'].index(appraisal['id'])
                        stimulus['corr_appraisals_ids'][corr_idx] = idx + 1
                appraisal['id'] = idx + 1
        # 3. 调整stimuli的序号
        for idx, stimulus in enumerate(remained_stimuli_list):
            if stimulus['id'] != idx + 1:
                stimulus['id'] = idx + 1
        # --------------------------------------------------

        # --------------- 四、用筛选和修改后的数据重新构建链条 ----------------
        postprocess_generation = ''
        if theme:
            postprocess_generation += "Theme:\n{}\n".format(theme)
        else:
            postprocess_generation += "Theme:\n{}\n".format('None.')
        postprocess_generation += 'Reactions of {} in #{}:\n'.format(target_speaker, target_utt_id)
        if len(remained_reactions_list):
            for reaction in remained_reactions_list:
                postprocess_generation += "{}. {}\n".format(str(reaction['id']), reaction['content'])
        else:
            postprocess_generation += "None.\n"
        postprocess_generation += 'Appraisals of {} in #{}:\n'.format(target_speaker, target_utt_id)
        if len(remained_appraisals_list):
            for appraisal in remained_appraisals_list:
                postprocess_generation += "{}. For Reaction {}: {}\n".format(str(appraisal['id']), ' & '.join([str(i) for i in appraisal['corr_reactions_ids']]), appraisal['content'])
        else:
            postprocess_generation += "None.\n"
        postprocess_generation += "Stimuli:\n"
        if len(remained_stimuli_list):
            for stimulus in remained_stimuli_list:
                postprocess_generation += "{}. For Appraisal {}: {}\n".format(str(stimulus['id']), ' & '.join([str(i) for i in stimulus['corr_appraisals_ids']]), stimulus['content'])
        else:
            postprocess_generation += "None.\n"
        answer_pred = sorted(list(set(pred_stimuli) - set(delete_utt_ids)))
        wrong_pred = sorted(list(set(delete_utt_ids)))
        assert not set(label).intersection(set(wrong_pred))
        postprocess_generation += "Causal utterances:\n{}".format(str(answer_pred))
        if complete:
            assert set(answer_pred) == set(label) # label中元素的顺序不一定是从小到大的
        # --------------------------------------------------

        # --------------- 五、构建json字典 ----------------
        info_dict[dialog_id]['theme'] = theme if theme else 'None.'
        info_dict[dialog_id]['stimuli_list'] = remained_stimuli_list
        info_dict[dialog_id]['appraisals_list'] = remained_appraisals_list
        info_dict[dialog_id]['reactions_list'] = remained_reactions_list
        info_dict[dialog_id]['answer_label'] = label
        info_dict[dialog_id]['answer_pred'] = answer_pred
        info_dict[dialog_id]['wrong_pred'] = wrong_pred
        info_dict[dialog_id]['complete'] = complete
        info_dict[dialog_id]['generation'] = postprocess_generation.replace('\n', '\\n') # 在标准的JSON中，字符串内部不允许出现未转义的换行符
        # --------------------------------------------------

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    # ---- set before run ----
    method_dir = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot'
    raw_generation_file = os.path.join(method_dir, '4shot_train', 'raw_generation_for_train.csv')
    save_path = os.path.join(method_dir, '4shot_train', 'postprocess.json')
    # ------------------------

    generation_dict = generation_extract(raw_generation_file)
    postprocess(generation_dict, save_path)
