import os
import csv
import pandas as pd
import numpy as np
import re
import sys
sys.path.append('/data4/hzp/ECR-Chain/chatgpt')
from get_prompt_from_template import *
from fill_template import fill_conversation_CEE_fewshot
from sklearn.metrics import classification_report

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

            # 后处理规则
            generation = generation.replace('`', '').strip() # 去除表示代码框的符号
            query = re.search("\[.*?\]", generation, re.I|re.M) # 找到满足条件（前后有中括号）的最短匹配 https://blog.csdn.net/weixin_42219542/article/details/109451826
            if query:
                query_str = query.group() # query.group()：得到字符串
            else:
                # 考虑生成的内容中无中括号包裹的情况，如果只缺了括号，也可以匹配一下格式
                if generation[0] != '[':
                    generation = '[' + generation
                if generation[-1] != ']':
                    generation = generation + ']'
                query_str = generation
            try:
                emo_utts = eval(query_str) # eval(): 将字符串形式转成列表形式
            except Exception as ex:
                emo_utts = []
        
            return_dicts[dialog_id] = {'pred_pos_utts': emo_utts}
    return return_dicts


def make_result_csv(result_dicts, save_path, full_set=True):
    '''
    用于人工分析case
    '''
    dialog_dicts = load_set('test')
    
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dialog_id', 'num_utt', 'conversation', 'target_utt', 'gt_pos_utts', 'pred_pos_utts'])

        if full_set:
            dialog_id_list = sorted(list(dialog_dicts.keys()))
        else:
            dialog_id_list = sorted(list(result_dicts.keys())) # 如果测试集只生成了部分数据，就用这句
        for dialog in dialog_id_list:
            dialog_dict = dialog_dicts[dialog]
            result_dict = result_dicts[dialog]
            num_utts = dialog_dict['num_utt']
            pred_pos_utts = result_dict['pred_pos_utts']
            writer.writerow([dialog, num_utts, fill_conversation_CEE_fewshot(dialog_dict), fill_targetutt_CEE_fewshot(dialog_dict), dialog_dict['pos_cause_utts'], pred_pos_utts])


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
    template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    template_name = 'CEE_fewshot'
    save_dir = os.path.join(template_root, template_name)
    raw_generation_path = os.path.join(save_dir, 'raw_generation.csv')
    save_path = os.path.join(save_dir, 'result.csv')

    result_dicts = answer_extraction(raw_generation_path)
    make_result_csv(result_dicts, save_path, full_set=False)
    reports_dict = cal_f1(save_path)
    print('neg_f1:{},\tpos_precision:{},\tpos_recall:{},\tpos_f1:{},\tmacro_f1:{}'.format(reports_dict['neg_f1'], reports_dict['pos_precision'], reports_dict['pos_recall'], reports_dict['pos_f1'], reports_dict['macro_f1']))