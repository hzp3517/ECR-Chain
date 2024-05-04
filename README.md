# ECR-Chain

The code for the paper "ECR-Chain: Advancing Generative Language Models to Better Emotion Cause Reasoners through Reasoning Chains" (IJCAI-2024).

The Appendix mentioned in the paper is present in [here](appendix.pdf).

The code of "Supervised CEE" part is based on [this version](https://github.com/lm-sys/FastChat/tree/8865b7c9e73c2ab84dafc8150b5040560bbe412f) of `lm-sys/FastChat`.


## Requirements
- python==3.10
- torch==1.13.1+cu116
- transformers==4.28.1
- fschat==0.2.15 
    - install from [source](https://github.com/lm-sys/FastChat/tree/8865b7c9e73c2ab84dafc8150b5040560bbe412f) following the [instructions](https://github.com/lm-sys/FastChat/tree/8865b7c9e73c2ab84dafc8150b5040560bbe412f#method-2-from-source).
- deepspeed==0.9.0
- peft==0.4.0.dev0
- openai==0.27.8


## Datasets

### RECCON-DD Dataset
The original RECCON dataset can be found at [this link](https://github.com/declare-lab/RECCON).

We preprocessed the original RECCON-DD dataset and saved on [RECCON_all_data.json](RECCON_all_data.json). Note that there are some repetitive positive causal pairs in the original dataset, which are removed in our preprocessing process, following the [KBCIN](https://github.com/circle-hit/KBCIN).


### ECR-Chain Set
Follow the code or instructions in `chatgpt/chatgpt_generation_for_construction.py` to perform the 4-stage process on the RECCON-DD training set and construct the ECR-Chain Set.

Our constructed ECR-Chain Set can be found at [here](chatgpt/prompt_examples/CEE_simplify_fewshot/simplify_subset.json).


## Few-shot CEE
Run the code `chatgpt/chatgpt_generation_for_fewshot_cee.py` to perform few-shot prompting with ECR-Chain on ChatGPT.

### Other Prompts for CEE
You need to modify `template_name` in the above code if use other prompts:

- Baseline Few-shot Prompt (directly predict the answer): `CEE_fewshot`
- ECR-Chain: `CEE_backward_v2_fewshot`
- ECR-Chain (without theme): `CEE_backward_v3_fewshot`
- ECR-Chain (without theme and reactions): `CEE_backward_v4_fewshot`
- ECR-Chain (without reactions): `CEE_backward_v5_fewshot`
- ECR-Chain (only with stimuli): `CEE_stimuli_fewshot`


## Supervised CEE
### Train Scripts
- Answer: `EmoReasonLlama/scripts/lora_train_answer.sh`
- Reasoning: `EmoReasonLlama/scripts/lora_train_reasoning.sh`
- Multi-task: `EmoReasonLlama/scripts/lora_train_multitask.sh`

### Inference Scripts
- Answer: `EmoReasonLlama/scripts/inference_answer.sh`
- Reasoning: `EmoReasonLlama/scripts/inference_reasoning.sh`

## Explainable ECR
Compare and analyze the quality of reasoning chains generated by two models via GPT-4. First save the two models' generated chains into json file and then get the GPT-4 analysis results via `gpt4_eval/gpt4_generation.py`.