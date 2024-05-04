import os
import csv
import pandas as pd
import numpy as np
import re
import copy
from sklearn.metrics import classification_report

class BaseEvaluate(object):
    def answer_extraction(self, generation):
        '''
        extract the task prediction for a single sample

        input:
        - `generation`: the model output without input prompt part (only the response part)

        return:
        - prediction
        '''
        raise NotImplementedError

    def batch_answer_extraction(self, batch_generation):
        '''
        extract the task prediction for a batch

        input:
        - `batch_generation`: the model output without input prompt part (only the response part)

        return:
        - batch_prediction
        '''
        raise NotImplementedError
    
    def compute_task_metrics(self, predictions, task_targets):
        '''
        input:
        - `predictions`: predictions of all samples on the target task
        - `task_targets`: ground truths of all samples on the target task
        '''
        raise NotImplementedError

class EvaluateForCEEVanilla(BaseEvaluate):
    def __init__(self):
        super(EvaluateForCEEVanilla, self).__init__()

    def answer_extraction(self, generation, num_utts):
        '''
        return:
        - prediction (list)
        '''
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

        # 检查并过滤列表：列表中的数字有可能不在合法范围内，即[1, num_utts] 且 为整数（上面的`eval()`已经保证列表中每个元素只会是数字）
        filter_emo_utts = []
        for n in emo_utts:
            if isinstance(n, int):
                if n >= 1 and n <= num_utts:
                    filter_emo_utts.append(n)
        filter_emo_utts = sorted(list(set(filter_emo_utts))) # 去重

        return filter_emo_utts

    def cal_task_metrics(self, predictions, task_targets, num_utts_list):
        '''
        input:
        - predictions & task_targets: list of list.
        - num_utts_list: list. Each item represent the number of utterences in each sample
        '''
        assert len(predictions) == len(task_targets) and len(predictions) == len(num_utts_list)

        final_onehot_preds = []
        final_onehot_targets = []

        for i in range(len(num_utts_list)):
            onehot_pred = np.zeros(num_utts_list[i], dtype=np.int32)
            onehot_pred[np.array(predictions[i], dtype=np.int32) - 1] = 1
            final_onehot_preds.append(onehot_pred)
            onehot_target = np.zeros(num_utts_list[i], dtype=np.int32)
            onehot_target[np.array(task_targets[i], dtype=np.int32) - 1] = 1
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


class EvaluateForCEEAnswer(BaseEvaluate):
    def __init__(self):
        super(EvaluateForCEEAnswer, self).__init__()

    def answer_extraction(self, generation, num_utts):
        '''
        return:
        - prediction (list)
        '''
        generation = generation.replace('`', '').strip() # 去除表示代码框的符号
        pos_pred_utts = []
        for i in range(1, num_utts + 1):
            if '#{}:'.format(str(i)) in generation:
                tmp_str = generation.split('#{}:'.format(str(i)))[-1]
                if len(tmp_str):
                    tmp_str = tmp_str.split('\n')[0]
                    if len(tmp_str):
                        tmp_str = tmp_str.strip()
                if 'Positive' in tmp_str:
                    pos_pred_utts.append(i)
                elif 'Negative' in tmp_str:
                    continue
                else:
                    continue
            else:
                continue
        return pos_pred_utts

    def cal_task_metrics(self, predictions, task_targets, num_utts_list):
        '''
        input:
        - predictions & task_targets: list of list.
        - num_utts_list: list. Each item represent the number of utterences in each sample
        '''
        assert len(predictions) == len(task_targets) and len(predictions) == len(num_utts_list)

        final_onehot_preds = []
        final_onehot_targets = []

        for i in range(len(num_utts_list)):
            onehot_pred = np.zeros(num_utts_list[i], dtype=np.int32)
            onehot_pred[np.array(predictions[i], dtype=np.int32) - 1] = 1
            final_onehot_preds.append(onehot_pred)
            onehot_target = np.zeros(num_utts_list[i], dtype=np.int32)
            onehot_target[np.array(task_targets[i], dtype=np.int32) - 1] = 1
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


class EvaluateForCEEBackward(BaseEvaluate):
    def __init__(self):
        super(EvaluateForCEEBackward, self).__init__()

    def answer_extraction(self, generation, num_utts):
        '''
        return:
        - prediction (list)
        '''
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

        # 检查并过滤列表：列表中的数字有可能不在合法范围内，即[1, num_utts] 且 为整数（上面的`eval()`已经保证列表中每个元素只会是数字）
        filter_emo_utts = []
        for n in emo_utts:
            if isinstance(n, int):
                if n >= 1 and n <= num_utts:
                    filter_emo_utts.append(n)
        filter_emo_utts = sorted(list(set(filter_emo_utts))) # 去重

        return filter_emo_utts

    def cal_task_metrics(self, predictions, task_targets, num_utts_list):
        '''
        input:
        - predictions & task_targets: list of list.
        - num_utts_list: list. Each item represent the number of utterences in each sample
        '''
        assert len(predictions) == len(task_targets) and len(predictions) == len(num_utts_list)

        final_onehot_preds = []
        final_onehot_targets = []

        for i in range(len(num_utts_list)):
            onehot_pred = np.zeros(num_utts_list[i], dtype=np.int32)
            onehot_pred[np.array(predictions[i], dtype=np.int32) - 1] = 1 # 经检验，即使predictions[i]为空列表，依然可以正确执行
            final_onehot_preds.append(onehot_pred)
            onehot_target = np.zeros(num_utts_list[i], dtype=np.int32)
            onehot_target[np.array(task_targets[i], dtype=np.int32) - 1] = 1
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



class EvaluateForEEVanilla(BaseEvaluate):
    def __init__(self):
        super(EvaluateForEEVanilla, self).__init__()

    def answer_extraction(self, generation):
        '''
        return:
        - prediction (list)
        '''
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
        return emo_utts

    def cal_task_metrics(self, predictions, task_targets, num_utts_list):
        '''
        input:
        - num_utts_list: list. Each item represent the number of utterences in each sample
        '''
        def cal_prf(pred_sids, gt_sids, num_utt_list):
            '''
            参考baseline代码中的函数修改
            '''
            pred_num, acc_num, true_num = 0, 0, 0
            for i in range(len(gt_sids)):
                for j in range(1, num_utt_list[i]+1):
                    if j in pred_sids[i]:
                        pred_num += 1
                    if j in gt_sids[i]:
                        true_num += 1
                    if j in pred_sids[i] and j in gt_sids[i]:
                        acc_num += 1
            p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
            f = 2*p*r/(p+r+1e-8)
            return p, r, f
        
        precision, recall, f1 = cal_prf(predictions, task_targets, num_utts_list)

        return dict(
            precision = precision,
            recall = recall,
            f1 = f1
        )


if __name__ == '__main__':
    # evaluate = EvaluateForCEEVanilla()
    # generation = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.3"
    # emo_utts = evaluate.answer_extraction(generation, num_utts=10)
    # print(emo_utts)

    evaluate = EvaluateForCEEAnswer()
    generation = "#1: Positive\n#2: Negative\n#3: Negative\n#4: hhh\n#5: Positivehahha"
    emo_utts = evaluate.answer_extraction(generation, num_utts=10)
    print(emo_utts)

