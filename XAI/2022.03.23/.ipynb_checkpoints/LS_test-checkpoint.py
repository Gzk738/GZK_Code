import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import nltk as tk
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig
import json
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# replace <PATd:/spofrte/modeH-TO-SAVED-MODEL> with the real path of the saved model
model_path = 'bert-large-uncased-whole-word-masking-finetuned-squad'

# load model
model = torch.load('D:\software/github/GZK_Code/XAI/2022.03.23/models/2022-03-24-00-14-43full_squad.pth')
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
# tokenizer = BertTokenizer.from_pretrained(model_path)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    output = model(inputs, token_type_ids=token_type_ids,
                   position_ids=position_ids, attention_mask=attention_mask, )
    return output.start_logits, output.end_logits


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values


ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
                    [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=token_type_ids,
                                                 position_ids=position_ids)

    return input_embeddings, ref_input_embeddings


def predict_qt(question, text):
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id,
                                                                cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    ground_truth = '13'

    start_scores, end_scores = predict(input_ids, \
                                       token_type_ids=token_type_ids, \
                                       position_ids=position_ids, \
                                       attention_mask=attention_mask)

    print('Question: ', question)
    print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]))
    return input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask, start_scores, end_scores, ground_truth, all_tokens,


def explain(input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask, start_scores, end_scores,
            ground_truth, all_tokens, ):
    lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

    attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(
                                                        token_type_ids, position_ids, attention_mask, 0),
                                                    internal_batch_size=4,
                                                    return_convergence_delta=True)
    attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                                additional_forward_args=(
                                                    token_type_ids, position_ids, attention_mask, 1),
                                                internal_batch_size=4,
                                                return_convergence_delta=True)

    attributions_start_sum = summarize_attributions(attributions_start)
    attributions_end_sum = summarize_attributions(attributions_end)
    # storing couple samples in an array for visualization purposes
    start_position_vis = viz.VisualizationDataRecord(
        attributions_start_sum,
        torch.max(torch.softmax(start_scores[0], dim=0)),
        torch.argmax(start_scores),
        torch.argmax(start_scores),
        str(ground_truth),
        attributions_start_sum.sum(),
        all_tokens,
        delta_start)

    end_position_vis = viz.VisualizationDataRecord(
        attributions_end_sum,
        torch.max(torch.softmax(end_scores[0], dim=0)),
        torch.argmax(end_scores),
        torch.argmax(end_scores),
        str(ground_truth),
        attributions_end_sum.sum(),
        all_tokens,
        delta_end)
    print(all_tokens)
    print('\033[1m', 'Visualizations For Start Position', '\033[0m')
    viz.visualize_text([start_position_vis])

    print('\033[1m', 'Visualizations For End Position', '\033[0m')

    print("attributions_start_sum:   ", len(attributions_start_sum))
    print("all tokens:    ", len(all_tokens))

    print(torch.max(torch.softmax(end_scores[0], dim=0)),
          torch.argmax(end_scores),
          torch.argmax(end_scores),
          torch.max(torch.softmax(start_scores[0], dim=0)))

    return all_tokens, attributions_start_sum


def pred_explain(question, text):
    input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask, start_scores, end_scores, ground_truth, all_tokens, = predict_qt(
        question, text)

    all_tokens, attributions_start_sum = explain(input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask,
                                                 start_scores, end_scores, ground_truth, all_tokens, )

    return all_tokens, attributions_start_sum, start_scores, end_scores


def generate_explain(i, question, text, answers):
    exp = {}
    l_attributions = []
    l_start_score = []
    l_end_score = []
    l_pred = []

    tokens = tk.sent_tokenize(text)
    all_tokens, attributions_start_sum, start_scores, end_scores = pred_explain(question, text)
    end_score = float(torch.max(torch.softmax(end_scores[0], dim=0)))
    start_score = float(torch.max(torch.softmax(start_scores[0], dim=0)))
    l_attributions.append(torch.sum(attributions_start_sum) / len(all_tokens))
    l_start_score.append(start_score)
    l_end_score.append(end_score)
    l_pred.append(' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]))

    for token in tokens:
        all_tokens, attributions_start_sum, start_scores, end_scores = pred_explain(question, token)
        end_score = float(torch.max(torch.softmax(end_scores[0], dim=0)))
        start_score = float(torch.max(torch.softmax(start_scores[0], dim=0)))
        l_attributions.append(torch.sum(attributions_start_sum) / len(all_tokens))
        l_start_score.append(start_score)
        l_end_score.append(end_score)
        l_pred.append(' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]))

    #?????????float
    l_attributions = [float(i) for i in l_attributions]

    exp["ids"] = i
    exp["preds"] = l_pred
    exp["attributions"] = l_attributions
    exp["start_score"] = l_start_score
    exp["end_score"] = l_end_score
    exp["answer"] = answers
    return exp
def write_explain(exp):
    exp_str = json.dumps(exp)
    with open(r"small_answers.txt", "a")as f:
        f.write(exp_str)
        f.write("\r\n")
    f.close()

def create_multi_bars(i, answers, labels, datas, tick_step=1, group_gap=0.2, bar_gap=0, ):
    '''
    labels : x?????????????????????
    datas ????????????????????????????????????????????????????????????????????????labels???????????????
    tick_step ?????????x??????????????????1?????????tick_step?????????x??????????????????
    group_gap : ??????????????????????????????????????????????????????????????????????????????
    bar_gap ??????????????????????????????????????????0??????????????????????????????????????????????????????????????????????????????????????????
    '''
    # ticks???x?????????
    ticks = np.arange(len(labels)) * tick_step
    # group_num???????????????????????????????????????????????????
    group_num = len(datas)
    # group_width??????????????????????????????group_gap ????????????????????????????????????
    group_width = tick_step - group_gap
    # bar_span????????????????????????x???????????????????????????????????????????????????
    bar_span = group_width / group_num
    # bar_width??????????????????????????????
    bar_width = bar_span - bar_gap
    # baseline_x???????????????????????????????????????x???????????????????????????????????????bar_span??????
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        if index == 0:
            plt.bar(baseline_x + index*bar_span, y, bar_width, label='attribution score')
        if index == 1:
            plt.bar(baseline_x + index*bar_span, y, bar_width, label='forword attention score')
        if index == 2:
            plt.bar(baseline_x + index*bar_span, y, bar_width, label='backword attention score')
    plt.ylabel('Scores')
    plt.title('LONG SHORT TEST')
    # x????????????????????????x???????????????
    plt.xticks(ticks, labels)
    plt.legend()

    plt.savefig('./longshorttest/' + str(i)+ '.jpg')
    #????????????
    plt.clf()


def generate_directory(i):
    folder = "./longshorttest/" + str(i)
    # ?????????py???????????????????????????????????????new_folder???????????????test?????????
    save_path = folder  # './output/Modify_algorithm'
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    except OSError:
        print('Error: Creating directory. ' + save_path)
def write_image(i, exp, answers):
    li = []
    # exp["preds"] = l_pred
    # exp["attributions"] = l_attributions
    # exp["start_score"] = l_start_score
    # exp["end_score"] = l_end_score
    l_attributions = exp["attributions"]
    l_start_score = exp["start_score"]
    l_end_score = exp["end_score"]

    for attribution in l_attributions:
        temp = attribution
        li.append(temp)
    l_attributions = li

    l_attributions = [i*6 for i in l_attributions]

    generate_directory(i)
    label = []
    for attribution in range(len(l_attributions)):
        label.append("sen" + str(attribution))

    data = [l_attributions, l_start_score, l_end_score]
    create_multi_bars(i, answers, label, data, bar_gap=0.1)

def write_exp(i, exp,answers):
    write_explain(exp)
    write_image(i, exp, answers)

from datasets import load_dataset

datasets = load_dataset('squad')

for i in range(len(datasets['validation'])):
    text = datasets['validation'][i]['context']
    question = datasets['validation'][i]['question']
    answers = datasets['validation'][i]['answers']['text'][0]

    exp = generate_explain(i, question, text, answers)

    write_exp(i, exp, answers)



