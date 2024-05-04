import os
import csv
import pandas as pd
import numpy as np
import re
import sys
sys.path.append('/data4/hzp/ECR-Chain/chatgpt')
from get_prompt_from_template import *
from fill_template import fill_conversation_CEE_backward_fewshot
from sklearn.metrics import classification_report
import copy

def answer_extraction(raw_generation_path):
    '''
    从初步生成的csv文件中以一定格式提取出结果。
    需要先分析生成的情况然后设置一些后处理规则
    '''
    return_dicts = {}

    with open(raw_generation_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            dialog_id = row['dialog_id']
            generation = row['generation']
            raw_generation = copy.deepcopy(generation)

            # 后处理规则
            generation = generation.strip().split('Causal utterances:')[-1] # 找到"Causal utterances:"后面的部分
            query = re.search("\[.*?\]", generation, re.I|re.M) # 找到满足条件（前后有中括号）的最短匹配 https://blog.csdn.net/weixin_42219542/article/details/109451826
            if query:
                query_str = query.group() # query.group()：得到字符串
            else:
                query_str = ''
            try:
                emo_utts = eval(query_str) # eval(): 将字符串形式转成列表形式
            except Exception as ex:
                emo_utts = []
            emo_utts = sorted(list(set(emo_utts))) # 去重操作
        
            return_dicts[dialog_id] = {'pred_pos_utts': emo_utts, 'generation': raw_generation}
    return return_dicts


def make_result_csv(result_dicts, save_path, full_set=True):
    '''
    用于人工分析case
    '''
    dialog_dicts = load_set('test')
    
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dialog_id', 'num_utt', 'conversation', 'target_utt', 'gt_pos_utts', 'pred_pos_utts', 'generation'])

        if full_set:
            dialog_id_list = sorted(list(dialog_dicts.keys()))
        else:
            dialog_id_list = sorted(list(result_dicts.keys())) # 如果测试集只生成了部分数据，就用这句
        for dialog in dialog_id_list:
            dialog_dict = dialog_dicts[dialog]
            result_dict = result_dicts[dialog]
            generation = result_dict['generation']
            num_utts = dialog_dict['num_utt']
            pred_pos_utts = result_dict['pred_pos_utts']
            writer.writerow([dialog, num_utts, fill_conversation_CEE_fewshot(dialog_dict), fill_targetutt_CEE_fewshot(dialog_dict), dialog_dict['pos_cause_utts'], pred_pos_utts, generation])


def cal_f1(result_file):
    '''
    计算pos_f1, neg_f1, macro_f1
    '''
    csv_data = pd.read_csv(result_file)
    num_utts = list(csv_data['num_utt'])
    gt_pos_utts = [eval(i) for i in csv_data['gt_pos_utts']] # 从csv中直接读出的每个元素是str类型，需要转换为list
    pred_pos_utts = [eval(i) for i in csv_data['pred_pos_utts']] # 从csv中直接读出的每个元素是str类型，需要转换为list

    assert len(pred_pos_utts) == len(gt_pos_utts) and len(pred_pos_utts) == len(num_utts)

    final_onehot_preds = []
    final_onehot_targets = []

    for i in range(len(num_utts)):
        onehot_pred = np.zeros(num_utts[i], dtype=np.int32)
        onehot_pred[np.array(pred_pos_utts[i], dtype=np.int32) - 1] = 1
        final_onehot_preds.append(onehot_pred)
        onehot_target = np.zeros(num_utts[i], dtype=np.int32)
        onehot_target[np.array(gt_pos_utts[i], dtype=np.int32) - 1] = 1
        final_onehot_targets.append(onehot_target)

    final_onehot_preds = list(np.concatenate(final_onehot_preds))
    final_onehot_targets = list(np.concatenate(final_onehot_targets))

    reports = classification_report(
        final_onehot_targets,
        final_onehot_preds,
        target_names=['neg', 'pos'],
        digits=4,
        output_dict=True
    )

    return dict(
        neg_f1 = reports['neg']['f1-score'],
        pos_f1 = reports['pos']['f1-score'],
        macro_f1 = reports['macro avg']['f1-score'],
        pos_precision = reports['pos']['precision'],
        pos_recall = reports['pos']['recall'],
    )


if __name__ == '__main__':
    # # ------- evaluate for full set --------------------
    # template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    # template_name = 'CEE_backward_v5_fewshot'
    # save_dir = os.path.join(template_root, template_name)
    # raw_generation_path = os.path.join(save_dir, 'raw_generation.csv')
    # save_path = os.path.join(save_dir, 'result.csv')

    # result_dicts = answer_extraction(raw_generation_path)
    # make_result_csv(result_dicts, save_path) # 生成结果文件
    # reports_dict = cal_f1(save_path) # 基于生成的结果文件计算评价指标
    # print('neg_f1:{},\tpos_f1:{},\tmacro_f1:{}'.format(reports_dict['neg_f1'], reports_dict['pos_f1'], reports_dict['macro_f1']))


    # ------- evaluate for selected set --------------------
    template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    template_name = 'CEE_backward_v5_fewshot'
    save_dir = os.path.join(template_root, template_name)
    raw_generation_path = os.path.join(save_dir, '4shot_selected_results', 'raw_generation.csv')
    save_path = os.path.join(save_dir, '4shot_selected_results', 'result.csv')

    result_dicts = answer_extraction(raw_generation_path)
    make_result_csv(result_dicts, save_path, full_set=False) # 生成结果文件
    reports_dict = cal_f1(save_path) # 基于生成的结果文件计算评价指标
    print('neg_f1:{},\tpos_precision:{},\tpos_recall:{},\tpos_f1:{},\tmacro_f1:{}'.format(reports_dict['neg_f1'], reports_dict['pos_precision'], reports_dict['pos_recall'], reports_dict['pos_f1'], reports_dict['macro_f1']))