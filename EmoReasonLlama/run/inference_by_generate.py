'''
利用`model.generate()`得到模型的输出结果并计算评测指标
'''
import argparse
from typing import Optional
import os
import torch
import transformers
from transformers import GenerationConfig
from peft import PeftModel
from tqdm import tqdm
import json

from EmoReasonLlama.data import *
from EmoReasonLlama.model.model_adapter import add_model_args, load_model
from EmoReasonLlama.modules.gptq import GptqConfig
from EmoReasonLlama.utils import is_partial_stop
from EmoReasonLlama.run.evaluation import *

@torch.inference_mode()
def get_model_generation(dataset, model, device, generation_config, max_new_tokens):
    ''' 
    return: 
    - list of dict: [{'input_id', 'label', 'generation'}, ...]
    '''
    return_list = []

    for sample in tqdm(list(dataset)):
        # sample_id = str(sample['dialog_id'])
        sample_idx = sample['dialog_idx']
        sample_id = dataset.dialog_ids[sample_idx]
        question_token_len = sample['question_token_len']
        question_text = dataset.tokenizer.decode(sample['input_ids'][:question_token_len], skip_special_tokens=True)

        question_input_ids = sample['input_ids'][:question_token_len].unsqueeze(0).to(device) # (seq_len,) -> (bs, seq_len)
        task_targets = sample['task_targets'].tolist()
        num_utts = sample['num_utts']
        generate_ids = model.generate( # `input_ids`: (bs, seq_len)
            input_ids=question_input_ids,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
        # full_output_text = dataset.tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False) # include the input prompt part
        # full_output_text = dataset.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] # for batch inference
        output_text = dataset.tokenizer.decode(generate_ids[0][question_token_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return_list.append({
            'dialog_id': sample_id,
            'question': question_text,
            'gt_answer': task_targets,
            'num_utts': num_utts,
            'generation': output_text
        })

    return return_list


def evaluate_for_emo_reason(dataset, model, device, generation_config, max_new_tokens):
    gene_dict_list = get_model_generation(dataset, model, device, generation_config, max_new_tokens)
    sample_id, questions, gt_answers, num_utts_list, generations = map(list, zip(*[d.values() for d in gene_dict_list]))

    if dataset.task_templ_type == 'CEE_vanilla' or dataset.task_templ_type == 'CEE_tra_answer':
        eval_object = EvaluateForCEEVanilla()
    elif dataset.task_templ_type.startswith('CEE_backward') or dataset.task_templ_type.startswith('CEE_simplify') or dataset.task_templ_type.startswith('CEE_stimuli'):
        eval_object = EvaluateForCEEBackward()
    elif dataset.task_templ_type == 'CEE_answer':
        eval_object = EvaluateForCEEAnswer()
    else:
        raise Exception('Invalid task template name!')
    
    assert len(num_utts_list) == len(generations)
    task_predictions = [eval_object.answer_extraction(generations[i], num_utts_list[i]) for i in range(len(num_utts_list))]
    metrics_dict = eval_object.cal_task_metrics(task_predictions, gt_answers, num_utts_list)

    return gene_dict_list, metrics_dict


def save_predictions_and_metrics(set_name, model_path, gene_dict_list, metrics_dict, save_predictions, save_metrics, manually_set_save_dir=None):
    '''
    如果直接用base model做inference，model_path的值为None，此时需要手动指定manually_set_save_dir，并按该路径保存inference结果
    '''
    if save_predictions:
        if model_path:
            if manually_set_save_dir:
                save_dir = manually_set_save_dir
            else:
                save_dir = os.path.join(model_path, 'inference_results')
        else:
            assert manually_set_save_dir
            save_dir = manually_set_save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        prediction_dict = {}
        for sample in gene_dict_list:
            prediction_dict[sample['dialog_id']] = {}
            prediction_dict[sample['dialog_id']]['gt'] = sample['gt_answer']
            prediction_dict[sample['dialog_id']]['pred'] = sample['generation']
        save_path = os.path.join(save_dir, '{}_predictions.json'.format(set_name))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(prediction_dict, f, ensure_ascii=False)
    if save_metrics:
        if model_path:
            if manually_set_save_dir:
                save_dir = manually_set_save_dir
            else:
                save_dir = os.path.join(model_path, 'inference_results')
        else:
            assert manually_set_save_dir
            save_dir = manually_set_save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, '{}_metrics.json'.format(set_name))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)

    # ----- params of data -----
    parser.add_argument("--dataset", type=str, default='CEECausalLMDataset', help='class name of the dataset')
    parser.add_argument("--set_name", type=str, default='test', choices=['train', 'dev', 'test', 'tra_valid_test'], help='inference on which set.')
    parser.add_argument("--raw_data_path", type=str, default='/data4/hzp/ECR-Chain/all_data.json', help='data path')
    parser.add_argument("--cot_gene_type", type=str, default='chatgpt', help='Where the cot knowledge from, [chatgpt, vicuna]')
    parser.add_argument("--cot_gene_dir", type=str, default='None', help='The inference_results dir of cot generation model if `cot_gene_type` is vicuna.')
    parser.add_argument("--conv_templ_type", type=str, default='vicuna_v1.1', choices=['alpaca', 'vicuna_v1.1'], help="specify a conversation template")
    # parser.add_argument("--task_templ_type", type=str, default='CEE_vanilla', help='specify task prompt and response template for the target task')
    parser.add_argument("--main_task_templ", type=str, default='CEE_vanilla', help='specify task prompt and response template for the target task')
    parser.add_argument("--example_id", type=str, default='tr_264_6', help='The sample id of the exemplar selected from the train set')
    parser.add_argument("--max_remain_utts", type=int, default=-1, help='Delete some utts early in the conversation to control the len of the overall input. -1 means not set.')

    # ----- params of tokenizer ------
    parser.add_argument("--padding_side", type=str, default='right', help='For batch inference, set it to `left` is better.')

    # ----- params of `model.generate()` ------
    parser.add_argument("--num_beams", type=int, default=1, help='Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search.')
    parser.add_argument("--temperature", type=float, default=1.0, help='The value used to module the next token probabilities. Must be strictly positive.') # 理论上0代表greedy decoding，但是generate()不支持等于0，可以设一个小量。
    parser.add_argument("--top_k", type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering')
    parser.add_argument("--top_p", type=float, default=1.0, help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
    parser.add_argument("--max_new_tokens", type=int, default=128, help='The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt')
    
    # ----- params of lora tuning ------
    parser.add_argument("--lora_path", type=str, default=None, help='if specified to some path, it will wrap the model with a lora weight')

    # ----- save or not -----
    parser.add_argument("--save_predictions", action='store_true', help='if specified, save the predictions into json files')
    parser.add_argument("--save_metrics", action='store_true', help='if specified, save the result dict into json files')
    parser.add_argument("--manually_set_save_dir", type=str, default=None, help='need to set if the `model-path` is not specified')

    args = parser.parse_args()

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # load the (base) model and the tokenizer
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        GptqConfig(
            ckpt=args.gptq_ckpt or args.model_path,
            wbits=args.gptq_wbits,
            groupsize=args.gptq_groupsize,
            act_order=args.gptq_act_order,
        ),
        args.revision,
        args.padding_side,
    )

    # if needed, load the lora weight
    if args.lora_path:
        if args.device == "cuda":
            model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                torch_dtype=torch.float16,
            )
        elif args.device == "mps":
            model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                device_map={"": args.device},
                torch_dtype=torch.float16,
            )
        else:
            model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                device_map={"": args.device},
            )

    device = args.device
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )
    max_new_tokens = args.max_new_tokens
    model_path = args.model_path
    save_predictions = args.save_predictions
    save_metrics = args.save_metrics
    if args.lora_path:
        manually_set_save_dir = os.path.join(args.lora_path, 'inference_results')
    if args.manually_set_save_dir:
        manually_set_save_dir = args.manually_set_save_dir

    if args.dataset == 'CEECausalLMDataset':
        dataset_class = CEECausalLMDataset
        eval_set = dataset_class(args.set_name, tokenizer, args.raw_data_path, args.conv_templ_type, args.main_task_templ)
    elif args.dataset == 'CEEBackwardFilterCausalLMDataset':
        dataset_class = CEEBackwardFilterCausalLMDataset
        eval_set = dataset_class(args.set_name, tokenizer, args.raw_data_path, None, args.conv_templ_type, args.main_task_templ, args.example_id, args.max_remain_utts, local_rank=0)
    elif args.dataset == 'CEEBackwardFilter0shotCausalLMDataset':
        dataset_class = CEEBackwardFilter0shotCausalLMDataset
        eval_set = dataset_class(args.set_name, tokenizer, args.raw_data_path, None, args.conv_templ_type, args.main_task_templ, args.max_remain_utts, local_rank=0)
    elif args.dataset == 'CEEStimuliCausalLMDataset':
        dataset_class = CEEStimuliCausalLMDataset
        eval_set = dataset_class(args.set_name, tokenizer, args.raw_data_path, None, args.conv_templ_type, args.main_task_templ, args.example_id, args.max_remain_utts, local_rank=0)
    elif args.dataset == 'CEETRAAnswerCausalLMDataset':
        dataset_class = CEETRAAnswerCausalLMDataset
        eval_set = dataset_class(args.set_name, tokenizer, args.raw_data_path, args.cot_gene_type, args.cot_gene_dir, args.conv_templ_type, args.main_task_templ, local_rank=0)
    else:
        raise Exception('Invalid dataset name!')

    gene_dict_list, metrics_dict = evaluate_for_emo_reason(eval_set, model, device, generation_config, max_new_tokens)

    set_name = eval_set.set_name
    save_predictions_and_metrics(set_name, model_path, gene_dict_list, metrics_dict, save_predictions, save_metrics, manually_set_save_dir)