import os
import numpy as np
import csv
import pandas as pd

# --------------------- CEE_vanilla for RECCON dataset ---------------------------------
def fill_conversation_CEE_vanilla(dialog_dict):
    content = ""
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content += "#{}: {} ({}): \"{}\"\n".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i])
    return content

def fill_targetutt_CEE_vanilla(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"\n".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_response_CEE_vanilla(dialog_dict):
    # content = str(list(dialog_dict['pos_cause_utts']))
    content = str(sorted(list(dialog_dict['pos_cause_utts'])))
    return content


# --------------------- CEE_simplify for RECCON dataset ---------------------------------
def fill_conversation_CEE_simplify(dialog_dict, max_remain_utts=-1):
    content_list = []
    num_utt = dialog_dict['num_utt']
    for i in range(num_utt):
        content_list.append("#{}: {} ({}): \"{}\"".format(str(i+1), dialog_dict['speaker_list'][i], dialog_dict['emotion_list'][i], dialog_dict['content_list'][i]))
    if max_remain_utts > 0: # 如果对话超过设定长度，需要把最前面的几条去掉
        del_num = num_utt - max_remain_utts
        if del_num > 0:
            content_list = content_list[del_num:]
            content_list = ['...'] + content_list # 去掉之后在最前面直接补个省略号
    content = '\n'.join(content_list)
    return content

def fill_targetutt_CEE_simplify(dialog_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    content = "#{}: {} ({}): \"{}\"".format(str(target_idx+1), dialog_dict['speaker_list'][target_idx], dialog_dict['emotion_list'][target_idx], dialog_dict['content_list'][target_idx])
    return content

def fill_exam_response_CEE_simplify(dialog_dict, rationales_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
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
    cause = "Causal utterances:\n{}".format(str(sorted(list(dialog_dict['pos_cause_utts']))))
    content = theme + reaction + appraisal + stimuli + cause
    return content

def fill_target_response_CEE_simplify(dialog_dict, rationales_dict):
    num_utt = dialog_dict['num_utt']
    target_idx = num_utt - 1
    sample_id = dialog_dict['sample_id']
    theme = "Theme:\n{}\n".format(rationales_dict[sample_id]['theme'])
    reaction = 'Reactions of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    if len(rationales_dict[sample_id]['reactions_list']):
        for i, reaction_item in enumerate(rationales_dict[sample_id]['reactions_list']):
            assert i+1 == reaction_item['id']
            reaction += "{}. {}\n".format(reaction_item['id'], reaction_item['content'])
    else:
        reaction += "None.\n"
    appraisal = 'Appraisals of {} in #{}:\n'.format(dialog_dict['speaker_list'][target_idx], target_idx + 1)
    if len(rationales_dict[sample_id]['appraisals_list']):
        for i, appr_item in enumerate(rationales_dict[sample_id]['appraisals_list']):
            assert i+1 == appr_item['id']
            appraisal += "{}. {}\n".format(appr_item['id'], appr_item['content'])
    else:
        appraisal += "None.\n"
    stimulus = 'Stimuli:\n'
    if len(rationales_dict[sample_id]['stimuli_list']):
        for i, sti_item in enumerate(rationales_dict[sample_id]['stimuli_list']):
            assert i+1 == sti_item['id']
            stimulus += "{}. {}\n".format(sti_item['id'], sti_item['content'])
    else:
        stimulus += "None.\n"
    cause = "Causal utterances:\n{}".format(str(sorted(list(dialog_dict['pos_cause_utts']))))
    content = theme + reaction + appraisal + stimulus + cause
    return content