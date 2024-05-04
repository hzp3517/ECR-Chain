import openai
import os
from tqdm import tqdm
from get_prompt_from_template import *
import pandas as pd
import csv

# -----You may need to add if using a proxy-----
# os.environ["http_proxy"] = "http://localhost:1081" 
# os.environ["https_proxy"] = "http://localhost:1081"
# -------------------------

openai.api_key = "sk-......" # Replace with your OpenAI key

def get_completion(prompt, model="gpt-3.5-turbo-0613"):
    messages = [{"role": "user", "content": prompt}]
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

def save_raw_generation_1st_iter(prompt_dict, save_path, err_path):
    '''
    Save the generation result in csv file.
    columns: `dialog_id`, `generation`

    If the execution fails, save the error message and dialog id to `err.csv`
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
        success, output = get_completion(prompt_dict[dialog_id])
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

def afterwards_iter(prompt_dict, save_path, err_path):
    last_err_df = pd.read_csv(err_path, header=None)
    last_err_ids = [str(i).zfill(4) for i in list(last_err_df[0])]

    lines = []
    err_idx_dict = {}
    f = open(save_path, "r+")
    reader = csv.reader(f)
    line = next(reader) # skip the head line
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
        success, output = get_completion(prompt_dict[dialog_id])
        if success:
            lines[err_idx_dict[dialog_id]] = [dialog_id, output]
        else:
            new_err_ids.append(dialog_id)
            new_err_info.append(output)
    
    f.seek(0) # The file pointer must be reset to zero before writing
    writer = csv.writer(f)
    writer.writerows(lines)
    f.close()

    new_err_df = pd.DataFrame({'err_ids': new_err_ids, 'err_info': new_err_info})
    new_err_df.to_csv(err_path, header=None, index=None)

    print('{} error items after this iteration: {}'.format(len(new_err_ids), new_err_ids))
    return new_err_ids


if __name__ == '__main__':
    ###########################################
    # Step 1: "Reasoning" via ChatGPT
    ###########################################
    template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    template_name = 'CEE_backward_v2_fewshot'
    exam_dia_ids = ['tr_264_6', 'tr_5811_4', 'tr_2553_8', 'tr_3449_3'] # 4-shot
    raw_generation_save_path = os.path.join(template_root, template_name, 'raw_generation_for_train.csv')
    err_path = os.path.join(template_root, template_name, 'err_for_train.csv')
    prompt_template = get_template(template_name, exam_dia_ids)
    prompt_dict = get_testset_prompt(template_name, prompt_template, test_set='train')
    err_ids = save_raw_generation_1st_iter(prompt_dict, raw_generation_save_path, err_path)
    while len(err_ids):
        err_ids = afterwards_iter(prompt_dict, raw_generation_save_path, err_path)

    
    ###########################################
    # Step 2: Postprocess of "Reasoning" (include the "Filtering" process)
    ###########################################
    # run `chatgpt/make_rationale_trainset/postprocess_of_backward_cot.py`


    ###########################################
    # Step 3: "Rationalization" via ChatGPT
    ###########################################
    # template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    # template_name = 'CEE_supp_fewshot'
    # exam_dia_ids = ['tr_1287_9_cause6', 'tr_3676_5_cause3', 'tr_20_8_cause8', 'tr_2232_4_cause3'] # 4-shot
    # raw_generation_save_path = os.path.join(template_root, template_name, 'raw_generation_for_rationalization.csv')
    # err_path = os.path.join(template_root, template_name, 'err_for_rationalization.csv')
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt_dict = get_testset_prompt(template_name, prompt_template, test_set='train')
    # err_ids = save_raw_generation_1st_iter(prompt_dict, raw_generation_save_path, err_path)
    # while len(err_ids):
    #     err_ids = afterwards_iter(prompt_dict, raw_generation_save_path, err_path)


    ###########################################
    # Step 4: Postprocess of "Rationalization"
    ###########################################
    # run `chatgpt/make_rationale_trainset/postprocess_of_rationalization.py`


    ###########################################
    # Step 5: "Consolidation" via ChatGPT
    ###########################################
    # template_root = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples'
    # template_name = 'CEE_simplify_fewshot'
    # exam_dia_ids = ['tr_1319_4', 'tr_132_2', 'tr_1086_3'] # 3-shot
    # raw_generation_save_path = os.path.join(template_root, template_name, 'raw_generation_for_simplify.csv')
    # err_path = os.path.join(template_root, template_name, 'err_for_simplify.csv')
    # prompt_template = get_template(template_name, exam_dia_ids)
    # prompt_dict = get_testset_prompt(template_name, prompt_template, test_set='train')
    # err_ids = save_raw_generation_1st_iter(prompt_dict, raw_generation_save_path, err_path)
    # while len(err_ids):
    #     err_ids = afterwards_iter(prompt_dict, raw_generation_save_path, err_path)


    ###########################################
    # Step 6: Postprocess of "Consolidation" and get the ECR-Chain set
    ###########################################
    # run `chatgpt/make_rationale_trainset/postprocess_of_simplify.py`
