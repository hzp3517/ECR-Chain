'''
将所有测试集的sample_id分成值为0和值为1的
- 值为0：评测时assistant 1是chatgpt，assistant 2是vicuna；
- 值为1：评测时assistant 1是vicuna，assistant 2是chatgpt；
对selected_test和remained_test分别处理，确保每个小集合内都是一半一半的划分（保证任何一个vicuna和chatgpt的比较都是尽量公平的）
注：selected_test是测试集中的一小部分，remained_test是剩余的测试集数据。两者共同构成整个test集。
'''

import os
import csv
import pandas as pd
from tqdm import tqdm
import json
import copy
import re
import random

dataset_json = '/data4/hzp/ECR-Chain/RECCON_all_data.json'
save_path = '/data4/hzp/ECR-Chain/gpt4_eval/sample_assignment.json'

# with open(dataset_json, 'r', encoding="utf-8") as f:
#     raw_data_dict = json.load(f)

# return_dict = {}

# for set_name in ['selected_test', 'remained_test']:
#     set_dict = raw_data_dict[set_name]
#     return_dict[set_name] = {}
#     all_sample_ids = list(set_dict.keys())
#     random.shuffle(all_sample_ids)
#     sample_num = len(all_sample_ids)
#     split_idx = sample_num // 2
#     for sid in all_sample_ids[:split_idx]:
#         return_dict[set_name][sid] = 0
#     for sid in all_sample_ids[split_idx:]:
#         return_dict[set_name][sid] = 1
    
# with open(save_path, "w", encoding="utf-8") as f:
#     json.dump(return_dict, f, ensure_ascii=False)




#  后处理：增加一个'test'字典
new_save_path = '/data4/hzp/ECR-Chain/gpt4_eval/sample_assignment_new.json'

with open(dataset_json, 'r', encoding="utf-8") as f:
    raw_data_dict = json.load(f)
with open(save_path, "r", encoding="utf-8") as f:
    old_save_dict = json.load(f)

return_dict = old_save_dict

set_dict = raw_data_dict['test']
return_dict['test'] = {}
for k in sorted(list(set_dict.keys())):
    if k in return_dict['selected_test']:
        return_dict['test'][k] = return_dict['selected_test'][k]
    else:
        assert k in return_dict['remained_test']
        return_dict['test'][k] = return_dict['remained_test'][k]

with open(new_save_path, "w", encoding="utf-8") as f:
    json.dump(return_dict, f, ensure_ascii=False)

