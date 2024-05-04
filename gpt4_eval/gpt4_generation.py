import openai
import os
from tqdm import tqdm
from get_prompt_from_template import *
import pandas as pd
import csv
import time

# -----需要加上这部分-----
os.environ["http_proxy"] = "http://localhost:1081" 
os.environ["https_proxy"] = "http://localhost:1081"
# -------------------------

openai.api_key = "sk-......" # replace to your openai key

def get_completion(system, prompt, model="gpt-4-1106-preview"):
    messages = [{"role": "system", "content": system}, 
                {"role": "user", "content": prompt}]
    success = True
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        output = response.choices[0].message["content"]
    except Exception as ex:
        success = False
        output = ex
    return success, output

def save_raw_generation_1st_iter(system, prompt_dict, save_path, err_path):
    '''
    以csv格式存储chatgpt生成结果。
    列名：dialog_id, generation

    如果执行失败，把错误信息和对话id保存到err.csv中
    '''
    print('------------the first iter-----------------')

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dialog_id', 'generation'])
    f = open(err_path, 'w')
    f.close()

    dialog_id_list = sorted(list(prompt_dict.keys()))

    err_ids = []
    err_info = []

    for dialog_id in tqdm(dialog_id_list):
        print('process dialog {}'.format(dialog_id))
        success, output = get_completion(system, prompt_dict[dialog_id])
        write_content = None if not success else output
        with open(save_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([dialog_id, write_content])
        if not success:
            err_ids.append(dialog_id)
            err_info.append(output)
            with open(err_path, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([dialog_id, output])

    print('{} error items after first iteration: {}'.format(len(err_ids), err_ids))
    return err_ids

def afterwards_iter(system, prompt_dict, save_path, err_path):
    last_err_df = pd.read_csv(err_path, header=None)
    last_err_ids = [str(i).zfill(4) for i in list(last_err_df[0])] # pandas读入会自动把id转换为int形式

    lines = []
    err_idx_dict = {}
    f = open(save_path, "r+")
    reader = csv.reader(f)
    line = next(reader) #表头
    lines.append(line)
    row_id = 1
    for line in reader:
        lines.append(line)
        if line[0] in last_err_ids:
            err_idx_dict[line[0]] = row_id
        row_id += 1

    new_err_ids = []
    new_err_info = []
    print('------------afterwards iter-----------------')
    for dialog_id in tqdm(last_err_ids):
        print('process dialog {}'.format(dialog_id))
        success, output = get_completion(system, prompt_dict[dialog_id])
        if success:
            lines[err_idx_dict[dialog_id]] = [dialog_id, output]
        else:
            new_err_ids.append(dialog_id)
            new_err_info.append(output)
    
    f.seek(0)
    writer = csv.writer(f)
    writer.writerows(lines)
    f.close()

    new_err_df = pd.DataFrame({'err_ids': new_err_ids, 'err_info': new_err_info})
    new_err_df.to_csv(err_path, header=None, index=None)

    print('{} error items after this iteration: {}'.format(len(new_err_ids), new_err_ids))
    return new_err_ids


if __name__ == '__main__':
    # ###################################
    baseline_model_name = 'vanilla_vicuna' # model-1 name
    baseline_chain_path = '.../vanilla_vicuna_chain.json'

    eval_model_name = 'multitask_vicuna' # model-2 name
    eval_model_path = '.../multitask_vicuna_chain.json'

    simplify = False # 完整的测试集用False
    eval_type = 'full' # ['cause', 'chain', 'full']
    set_name = 'test' # ['selected_test', 'remained_test', 'test']
    # ##################################

    eval_root = '/data4/hzp/ECR-Chain/gpt4_eval'
    eval_results_dir = os.path.join(eval_root, 'eval_results')
    save_dir = os.path.join(eval_results_dir, 'baseline_{}'.format(baseline_model_name), '{}-{}-{}'.format(eval_model_name, set_name, eval_type))
    os.makedirs(save_dir, exist_ok=False)

    raw_generation_save_path = os.path.join(save_dir, 'raw_generation.csv')
    err_path = os.path.join(save_dir, 'err.csv')
    system, prompt_dict = get_template(baseline_chain_path, eval_model_path, eval_type, set_name, simplify)

    err_ids = save_raw_generation_1st_iter(system, prompt_dict, raw_generation_save_path, err_path)
    while len(err_ids):
        err_ids = afterwards_iter(system, prompt_dict, raw_generation_save_path, err_path)