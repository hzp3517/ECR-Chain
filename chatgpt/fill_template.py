import os
import numpy as np
import csv
import pandas as pd
import json

# --------------------- CEE ---------------------------------
def fill_conversation_CEE_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_fewshot(dialog_dict):
    content = str(list(dialog_dict['pos_cause_utts']))
    return content


# --------------------- CEE_v2 ---------------------------------
emo_noun2adj_map = {"neutral": "neutral", "happiness": "happy", "sadness": "sad", "anger": "angry", "fear": "fearful", "surprise": "surprised", "disgust": "disgusted"}

def fill_conversation_CEE_v2_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "In utterance #{}, the speaker \"{}\" is {}: \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], emo_noun2adj_map[dialog_dict['emotion_list'][i]], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_v2_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "In utterance #{}, the speaker \"{}\" is {}: \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], emo_noun2adj_map[dialog_dict['emotion_list'][target_idx]], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_v2_fewshot(dialog_dict):
    content = str(list(dialog_dict['pos_cause_utts']))
    return content


# --------------------- CEE_stimuli ---------------------------------
def fill_conversation_CEE_stimuli_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_stimuli_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_stimuli_fewshot(dialog_dict):
    CEE_stimuli_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_stimuli_fewshot/sample_rationales.json'
    with open(CEE_stimuli_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    stimuli = "Stimuli:\n"
    for stimulus in rationales_dict[sample_id]['stimuli']:
        stimuli += "- {}\n".format(stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = stimuli + cause
    return content


# --------------------- CEE_backward ---------------------------------
def fill_conversation_CEE_backward_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_backward_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_backward_fewshot(dialog_dict):
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_fewshot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
    reaction = 'Reaction:\n'
    for i, react in enumerate(rationales_dict[sample_id]['reaction']):
        reaction += "{}. {}\n".format(str(i+1), react)
    appraisal = 'Appraisal:\n'
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for stimulus in rationales_dict[sample_id]['stimuli']:
        stimuli += "- {}\n".format(stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = theme + reaction + appraisal + stimuli + cause
    return content


# --------------------- CEE_backward_v2 ---------------------------------
def fill_conversation_CEE_backward_v2_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_backward_v2_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_backward_v2_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v2_fewshot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, react in enumerate(rationales_dict[sample_id]['reaction']):
        reaction += "{}. {}\n".format(str(i+1), react)
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(rationales_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = theme + reaction + appraisal + stimuli + cause
    return content


# --------------------- CEE_backward_1shot ---------------------------------
def fill_conversation_CEE_backward_1shot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_backward_1shot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_backward_1shot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_1shot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, react in enumerate(rationales_dict[sample_id]['reaction']):
        reaction += "{}. {}\n".format(str(i+1), react)
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(rationales_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = theme + reaction + appraisal + stimuli + cause
    return content





# --------------------- CEE_backward_v3 ---------------------------------
def fill_conversation_CEE_backward_v3_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_backward_v3_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_backward_v3_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v3_fewshot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, react in enumerate(rationales_dict[sample_id]['reaction']):
        reaction += "{}. {}\n".format(str(i+1), react)
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(rationales_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = reaction + appraisal + stimuli + cause
    return content


# --------------------- CEE_backward_v4 ---------------------------------
def fill_conversation_CEE_backward_v4_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_backward_v4_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_backward_v4_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v4_fewshot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(rationales_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = appraisal + stimuli + cause
    return content


# --------------------- CEE_backward_v5 ---------------------------------
def fill_conversation_CEE_backward_v5_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_backward_v5_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_results_CEE_backward_v5_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_backward_v5_fewshot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(rationales_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    cause = "Causal utterances:\n{}\n".format(str(list(dialog_dict['pos_cause_utts'])))
    content = theme + appraisal + stimuli + cause
    return content


# --------------------- CEE_supp ---------------------------------
def fill_conversation_CEE_supp_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_supp_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_causeutt_CEE_supp_fewshot(dialog_dict):
    cause_id = dialog_dict['cause_utt_id']
    content = "#{}: {} ({}): \"{}\"\n".format(str(cause_id), dialog_dict['speaker_list'][cause_id-1], dialog_dict['emotion_list'][cause_id-1], dialog_dict['content_list'][cause_id-1])
    return content

def fill_results_CEE_supp_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    cause_id = dialog_dict['cause_utt_id']
    cause_idx = cause_id - 1
    CEE_supp_rationalizations_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_supp_fewshot/sample_rationalizations.json'
    with open(CEE_supp_rationalizations_file, 'r', encoding="utf-8") as f:
        rationalizations_dict = json.load(f)
    sample_id = '{}_cause{}'.format(dialog_dict['dialog_id'], str(dialog_dict['cause_utt_id']))
    explanation = "Explanation:\n{}\n".format(rationalizations_dict[sample_id]['explanation'])
    stimulus = "Stimulus in #{}:\n- {}\n".format(cause_idx + 1, rationalizations_dict[sample_id]['stimulus'])
    appraisal = "Appraisal of {} in #{}:\n- {}\n".format(dialog_dict['speaker_list'][target_idx], target_idx + 1, rationalizations_dict[sample_id]['appraisal'])
    reaction = "Reaction of {} in #{}:\n- {}\n".format(dialog_dict['speaker_list'][target_idx], target_idx + 1, rationalizations_dict[sample_id]['reaction'])
    content = explanation + stimulus + appraisal + reaction
    return content


# --------------------- CEE_neg ---------------------------------
def fill_conversation_CEE_neg_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_neg_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_canutt_CEE_neg_fewshot(dialog_dict):
    can_id = dialog_dict['can_utt_id']
    content = "#{}: {} ({}): \"{}\"\n".format(str(can_id), dialog_dict['speaker_list'][can_id-1], dialog_dict['emotion_list'][can_id-1], dialog_dict['content_list'][can_id-1])
    return content

def fill_results_CEE_neg_fewshot(dialog_dict):
    CEE_neg_samples_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_neg_fewshot/sample_neg_explanations.json'
    with open(CEE_neg_samples_file, 'r', encoding="utf-8") as f:
        neg_samples_dict = json.load(f)
    sample_id = '{}_can{}'.format(dialog_dict['dialog_id'], str(dialog_dict['can_utt_id']))
    explanation = "Explanation:\n{}\n".format(neg_samples_dict[sample_id]['explanation'])
    judgement = "Judgement:\n{}\n".format(neg_samples_dict[sample_id]['judgement'])
    content = explanation + judgement
    return content


# --------------------- CEE_forward ---------------------------------
def fill_conversation_CEE_forward_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_forward_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_causalutts_CEE_forward_fewshot(dialog_dict):
    pos_cause_utts = list(sorted([int(i) for i in dialog_dict['pos_cause_utts']]))
    causalutts = ['#'+str(i) for i in pos_cause_utts]
    return str(causalutts).replace('\'', '') + '\n'

def fill_results_CEE_forward_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_backward_rationales_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_forward_fewshot/sample_rationales.json'
    with open(CEE_backward_rationales_file, 'r', encoding="utf-8") as f:
        rationales_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(rationales_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(rationales_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, react in enumerate(rationales_dict[sample_id]['reaction']):
        reaction += "{}. {}\n".format(str(i+1), react)
    content = theme + stimuli + appraisal + reaction
    return content


# --------------------- CEE_simplify ---------------------------------
def fill_conversation_CEE_simplify_fewshot(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_simplify_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_causeutts_CEE_simplify_fewshot(dialog_dict):
    cause_ids = dialog_dict['cause_utt_ids']
    content = '[{}]\n'.format(', '.join(['#'+str(i) for i in cause_ids]))
    return content

def fill_original_chains_CEE_simplify_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, react_item in enumerate(dialog_dict['reactions_list']):
        reaction += "{}. {}\n".format(str(i+1), react_item['content'])
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr_item in enumerate(dialog_dict['appraisals_list']):
        appraisal += "{}. {}\n".format(str(i+1), appr_item['content'])
    stimuli = "Stimuli:\n"
    for i, stimulus_item in enumerate(dialog_dict['stimuli_list']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus_item['content'])
    content = reaction + appraisal + stimuli
    return content

def fill_results_CEE_simplify_fewshot(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    CEE_simplify_file = '/data4/hzp/ECR-Chain/chatgpt/prompt_examples/CEE_simplify_fewshot/sample_simplified_rationales.json'
    with open(CEE_simplify_file, 'r', encoding="utf-8") as f:
        simplify_dict = json.load(f)
    sample_id = dialog_dict['sample_id']
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, react in enumerate(simplify_dict[sample_id]['reaction']):
        reaction += "{}. {}\n".format(str(i+1), react)
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    for i, appr in enumerate(simplify_dict[sample_id]['appraisal']):
        appraisal += "{}. {}\n".format(str(i+1), appr)
    stimuli = "Stimuli:\n"
    for i, stimulus in enumerate(simplify_dict[sample_id]['stimuli']):
        stimuli += "{}. {}\n".format(str(i+1), stimulus)
    content = reaction + appraisal + stimuli
    return content