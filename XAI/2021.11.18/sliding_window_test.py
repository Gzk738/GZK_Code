import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import collections
import torch
import torch.nn as nn
from datasets import load_dataset
import difflib

from scipy.ndimage import gaussian_filter

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# replace <PATd:/spofrte/modeH-TO-SAVED-MODEL> with the real path of the saved model
model_path = 'bert-large-uncased-whole-word-masking-finetuned-squad'

# load model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.to(device)
model.eval()
model.zero_grad()

"""++++++++++++++++++这几个函数是计算f1 score 数值的，代码是抄的，千万不能改！+++++++++++++++++"""


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers


"""+++++++++++++++++++++++++++++++++++"""


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


# load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)


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


fig = plt.figure()
fig.set_size_inches(8, 6)

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

    #print('Question: ', question)
    #print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]))
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
    # print(all_tokens)
    #print('\033[1m', 'Visualizations For Start Position', '\033[0m')
    #viz.visualize_text([start_position_vis])

    #print('\033[1m', 'Visualizations For End Position', '\033[0m')

    #print("attributions_start_sum:   ", len(attributions_start_sum))
    # print("all tokens:    ", len(all_tokens))

    return all_tokens, attributions_start_sum


def get_posneg(all_tokens, attributions_start_sum):
    positive = []
    negative = []
    neutral = []
    for i, j in enumerate(attributions_start_sum):
        if j > 0:
            positive.append(i)
            # print('positive:',j)
        ##print(all_tokens[i])
        elif j < 0:
            negative.append(i)
            # print('negative:',j)
            # print(all_tokens[i])
        elif j == 0:
            neutral.append(i)

    s_pos = ''
    s_neg = ''

    # print(len(attributions_start_sum))
    # print(len(positive))
    # print(len(negative))

    for i in positive:
        s_pos += all_tokens[i] + ' '
    # print("positive :", s_pos)
    for i in negative:
        s_neg += all_tokens[i] + ' '
    # print("negative :", s_neg)
    return positive, negative, neutral


def separate_sentence(all_tokens):
    sentence = {}
    temp = []
    num = 0
    for i in range(len(all_tokens)):
        if all_tokens[i] == "," or all_tokens[i] == ".":
            temp.append(all_tokens[i])
            sentence[num] = temp
            temp = []
            num = num + 1
        elif all_tokens[i] == "[CLS]":
            temp.append(all_tokens[i])
            sentence[num] = temp
            temp = []
            num = num + 1
        elif all_tokens[i] == "[SEP]":
            sentence[num] = temp
            num = num + 1
            temp = [all_tokens[i]]
            sentence[num] = temp
            temp = []
            num = num + 1
        else:
            temp.append(all_tokens[i])
    return sentence
def gaussian_scores(scores):
    # 使用高斯滤波替换贡献:
    old_contri = []
    for i in scores.values():
        old_contri.append(i)

    gaussian_contri = gaussian_filter(old_contri, sigma=5)
    gaussin_scores = {}
    index = 0
    for i, j in scores.items():
        gaussin_scores[i] = gaussian_contri[index]
        index = index + 1
    return gaussin_scores

def get_sence_score(sentence, attributions_start_sum):
    weight = 0
    sum_weight = 0
    sentence_value = []
    delete_sentence = []
    for k, v in sentence.items():
        for i in v:
            sentence_value.append(i)
    scores = {}

    for i in range(len(attributions_start_sum)):
        try:
            scores[sentence_value[i]] = abs(attributions_start_sum[i].item())
        except:
            pass
    #scores = gaussian_scores(scores)
    for i, j in sentence.items():
        sum_weight = 0
        for word in j:
            sum_weight += scores[word]
        delete_sentence.append(sum_weight)
        # print(sum_weight)
    return delete_sentence
def get_sence_gaussianscore(sentence, attributions_start_sum):
    weight = 0
    sum_weight = 0
    sentence_value = []
    delete_sentence = []
    for k, v in sentence.items():
        for i in v:
            sentence_value.append(i)
    scores = {}

    for i in range(len(attributions_start_sum)):
        try:
            scores[sentence_value[i]] = attributions_start_sum[i].item()
        except:
            pass
    scores = gaussian_scores(scores)
    for i, j in sentence.items():
        sum_weight = 0
        for word in j:
            sum_weight += scores[word]
        delete_sentence.append(sum_weight)
        # print(sum_weight)
    return delete_sentence

def get_delete(sentence):
    weight = 0
    sum_weight = 0
    sentence_value = []
    delete_sentence = {}
    for k, v in sentence.items():
        # print(k,':',v)
        for i in v:
            sentence_value.append(i)
    # print(sentence_value)
    scores = {}
    # print(attributions_start_sum[0].item())

    for i in range(len(attributions_start_sum)):
        try:
            scores[sentence_value[i]] = attributions_start_sum[i].item()
        except:
            pass

    for i, j in sentence.items():
        sum_weight = 0
        for word in j:
            weight = 0

            sum_weight += scores[word]
            delete_sentence[i] = sum_weight
    return delete_sentence


def delete_sentence(sentence, li_delete_sentence):
    for i, j in sentence.items():
        if i in li_delete_sentence:
            sentence[i] = []
        else:
            pass
    return sentence


def rebuild_sentence(ori_sentence):
    rebuild_str = ""
    for i, j in ori_sentence.items():
        for word in j:
            rebuild_str += word
            rebuild_str += " "
    return rebuild_str


def pred_explain(question, text):
    input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask, start_scores, end_scores, ground_truth, all_tokens, = predict_qt(question, text)

    all_tokens, attributions_start_sum = explain(input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask,
                                                 start_scores, end_scores, ground_truth, all_tokens, )

    end_score = float(torch.max(torch.softmax(end_scores[0], dim=0)))
    start_score = float(torch.max(torch.softmax(start_scores[0], dim=0)))

    return all_tokens, attributions_start_sum, end_score, start_score, [torch.argmax(start_scores), torch.argmax(
        end_scores) + 1], start_scores, end_scores


def max_min(x, y, z):
    max = min = x
    i = 1
    if y > max:
        max = y
        i = 2
    else:
        min = y
    if z > max:
        max = z
        i = 3
    else:
        min = z
    return (i)

def analysis(f1, acc_s, acc_e, sun):
    plt.plot(range(len(f1)), f1, "--bo", label="f1 score")
    plt.show()
    plt.plot(range(len(acc_s)), acc_s)
    plt.plot(range(len(acc_e)), acc_e)
    plt.plot(range(len(sun)), sun)
    plt.show()


def cycle_prediction(cycle_num, question, text, s_answer):
    all_tokens, attributions_start_sum, start_acc, end_acc, an_index, start_scores, end_scores = pred_explain(text,
                                                                                                              question)

    f1 = []
    acc_s = []
    acc_e = []
    sun = []
    ans = []
    second_answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
    second_answer = re.sub(r' ##', '', second_answer)
    f1_score = compute_f1(second_answer, s_answer)
    f1.append(f1_score)
    for loop in range(cycle_num):
        retext = rebuild_text(all_tokens, attributions_start_sum)

        all_tokens, attributions_start_sum, start_acc, end_acc, an_index, start_scores, end_scores = pred_explain(
            question, retext)
        reanswer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        # print(start_acc, end_acc)
        second_answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        second_answer = re.sub(r' ##', '', second_answer)
        # print("my answer is ", second_answer)
        ans.append(second_answer)
        # print(start_acc, end_acc)
        acc_s.append(start_acc)
        acc_e.append(end_acc)
        pos_contri = 0
        neg_contri = 0
        f1_score = compute_f1(second_answer, s_answer)
        f1.append(f1_score)

        # print(acc_s, acc_e)
        # print(acc_s, acc_e)

    for i in range(len(acc_s)):
        sun.append((acc_s[i] + acc_e[i]) / 2)
    return f1, acc_s, acc_e, sun
def cascading_rebuild_text(all_tokens, attributions_start_sum):

    li_sep = []
    min_sensocer = 999
    li_symbol = []
    min_index = 999
    sentence = separate_sentence(all_tokens)
    sentence_score = get_sence_score(sentence, attributions_start_sum)
    guassian_score = get_sence_gaussianscore(sentence, attributions_start_sum)

    for i in range(len(sentence_score)):
        if sentence_score[i] == 0:
            li_symbol.append(i)

    for i in li_symbol:
        guassian_score[i] == 0
    for i in range(len(guassian_score)):
        if sentence_score[i] < min_sensocer and sentence_score[i] != 0:
            min_sensocer = sentence_score[i]
            min_index = i
    print("should delete", min_index, min_sensocer)

    # temp = []
    # for i in sentence_score:
    #     temp.append(abs(i))
    # sentence[sentence_score.index(min(temp))] = ''


    sentence[min_index] = ''#删除贡献最小的句子
    sentence[1] = ''#删除问题
    retext = ""
    for i, j in sentence.items():
        for words in j:
            retext = retext + words + " "
    #这是清楚 ## 等模型引入的字符串
    for m in re.finditer(r"SEP", retext):
        li_sep.append(m.start())
        li_sep.append(m.end())
    retext = retext[li_sep[1] + 1: li_sep[2] - 1]
    retext = re.sub(r' ##', '', retext)
    return retext
def cascading_min(data, min_id):
    print("threshold : ", min_id)
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = 999
    d = {}
    li_min = []
    if min_id == 0:  # 设定一个空字典
        for i, v in enumerate(data):  # 利用函数enumerate列出lt的每个元素下标i和元素v
            d[v] = i  # 把v作为字典的键，v对应的值是i
        data.sort()  # 运用sort函数对lt元素排
        y = data[min_id]  # 此时lt中第二小的下标是1，求出对应的元素就是字典对应的键
        return [d[y]]
    if min_id != 0:
        for i, v in enumerate(data):  # 利用函数enumerate列出lt的每个元素下标i和元素v
            d[v] = i  # 把v作为字典的键，v对应的值是i
        data.sort()  # 运用sort函数对lt元素排
        for i in range(min_id):
            li_min.append(d[data[i]])  # 此时lt中第二小的下标是1，求出对应的元素就是字典对应的键
        return li_min



def independ_rebuild_text(all_tokens, attributions_start_sum, threshold):

    li_sep = []
    min_sensocer = 999
    li_symbol = []
    min_index = []
    sentence = separate_sentence(all_tokens)
    sentence_score = get_sence_score(sentence, attributions_start_sum)

    min_index = cascading_min(sentence_score, threshold)
    print("should delete : ", min_index)


    # temp = []
    # for i in sentence_score:
    #     temp.append(abs(i))
    # sentence[sentence_score.index(min(temp))] = ''

    for i in min_index:
        sentence[i] = ""#删除贡献最小的句子
    sentence[1] = ''#删除问题
    retext = ""
    for i, j in sentence.items():
        for words in j:
            retext = retext + words + " "
    #这是清楚 ## 等模型引入的字符串
    for m in re.finditer(r"SEP", retext):
        li_sep.append(m.start())
        li_sep.append(m.end())
    retext = retext[li_sep[1] + 1: li_sep[2] - 1]
    retext = re.sub(r' ##', '', retext)
    return retext
def muti_pre(cycle_num, question, text, s_answer, pro_keep, pro_next):
    all_tokens, attributions_start_sum, start_acc, end_acc, an_index, start_scores, end_scores = pred_explain(question, text)

    first_answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
    first_answer = re.sub(r' ##', '', first_answer)
    print("my answer is ", first_answer)
    ans = []
    acc_s= []
    acc_e = []
    f1 = []
    sun = []
    f1_score = compute_f1(first_answer, s_answer)
    f1.append(f1_score)
    acc_s.append(start_acc)
    acc_e.append(end_acc)
    sun.append((start_acc+end_acc)/2)
    for loop in range(cycle_num):
        retext = independ_rebuild_text(all_tokens, attributions_start_sum, loop)


        tokens, attributions, start_acc, end_acc, an_index, start_scores, end_scores = pred_explain(
            question, retext)

        second_answer = ' '.join(tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        second_answer = re.sub(r' ##', '', second_answer)
        ans.append(second_answer)
        acc_s.append(start_acc)
        acc_e.append(end_acc)

        f1_score = compute_f1(second_answer, s_answer)
        f1.append(f1_score)
        sun.append((start_acc + end_acc) / 2)
    return f1, acc_s, acc_e, sun



datasets = load_dataset('squad')
g_f1 = []
g_accs = []
g_acce = []
g_sun = []
handle = {}
wrong_ids = []
for i in range(len(datasets['validation'])):
    try:
        text = datasets['validation'][i]['context']
        question = datasets['validation'][i]['question']
        answers = datasets['validation'][i]['answers']
        f1, acc_s, acc_e, sun = muti_pre(10, question, text, answers['text'][0], 0.9, 0.7)
        print(i, "个完成了")
        g_f1.append(f1)
        g_accs.append(acc_s)
        g_acce.append(acc_e)
        g_sun.append(sun)
    except:
        print(i, "个失败")


#print(g_f1, g_accs ,g_acce ,g_sun )
handle["f1"] = g_f1
handle["forword_pro"] = g_accs
handle["backword_pro"] = g_acce
handle["sun_pro"] = g_sun
print(handle)


