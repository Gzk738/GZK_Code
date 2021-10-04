import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
from datasets import load_dataset
import difflib

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
            scores[sentence_value[i]] = attributions_start_sum[i].item()
        except:
            pass

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
    input_ids, ref_input_ids, token_type_ids, position_ids, attention_mask, start_scores, end_scores, ground_truth, all_tokens, = predict_qt(
        text, question)

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
        sentence = separate_sentence(all_tokens)
        sentence_score = get_sence_score(sentence, attributions_start_sum)
        min_sensocer = 999
        """min_index = 999
        for i in range(len(sentence_score)):
            if sentence_score[i] < min_sensocer and sentence_score[i] != 0:
                min_sensocer = sentence_score[i]
                min_index = i"""
        close20_index = 999
        for i in range(len(sentence_score)):
            if abs(sentence_score[i]) < abs(min_sensocer) and sentence_score[i] != 0:
                min_sensocer = sentence_score[i]
                close20_index = i
        # print("should delete", min_index, min_sensocer)
        sentence[close20_index] = ''
        sentence[1] = ''
        retext = ""
        for i, j in sentence.items():
            for words in j:
                retext = retext + words + " "
        li_sep = []
        for m in re.finditer(r"SEP", retext):
            li_sep.append(m.start())
            li_sep.append(m.end())
        retext = retext[li_sep[1] + 1: li_sep[2] - 1]
        retext = re.sub(r' ##', '', retext)

        all_tokens, attributions_start_sum, start_acc, end_acc, an_index, start_scores, end_scores = pred_explain(
            retext, question)
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
    """输出曲线"""
    """plt.plot(range(len(acc_s)), acc_s, label='start score')
    plt.plot(range(len(acc_s)), acc_e, label='end score')
    sun = []
    for i in range(len(acc_s)):
        sun.append((acc_s[i] + acc_e[i]) / 2)
    print(sun)
    plt.plot(range(len(acc_s)), sun, label='average')
    plt.xlabel('Number of predictions')
    plt.ylabel('Possibility')
    plt.legend()
    plt.show()"""

    """"获取最好的曲线并输出"""
    """max_start = 0
    max_end = 0
    max_ave = 0
    for i in acc_s:
        if i > max_start:
            max_start = i
    for j in acc_e:
        if j > max_end:
            max_end = i

    for x in sun:
        if x > max_ave:
            max_ave = x

    print(max_start, max_end, max_ave)

    max_list = max_min(max_start, max_end, max_ave)
    if max_list == 1:
        plt.plot(range(len(acc_s)), acc_s, label='Possibility')
        print(acc_s)
    if max_list == 2:
        plt.plot(range(len(acc_e)), acc_e, label='Possibility')
        print(acc_e)
    if max_list == 3:
        plt.plot(range(len(sun)), sun, label='Possibility')
        print(sun)

    plt.xlabel('Number of predictions')
    plt.ylabel('Possibility')
    plt.legend()
    plt.show()
    for i in range(len(ans)):
        print(ans[i])"""
    #输出score
    """plt.plot(range(len(f1)), f1, label='f1 score')
    plt.xlabel('Number of predictions')
    plt.ylabel('f1 score')
    plt.legend()
    plt.show()"""
    for i in range(len(acc_s)):
        sun.append((acc_s[i] + acc_e[i]) / 2)
    return f1, acc_s, acc_e, sun

incorrect_answer_id = [3, 23, 28, 30, 33, 39, 40, 41, 42, 43, 48, 49, 50, 53, 56, 58, 73, 74, 77, 84, 86, 96, 97, 329, 330, 336, 338, 344, 346, 349, 360, 364, 366, 370, 376, 379, 392, 393, 398, 415, 423, 424, 427, 430, 431, 437, 439, 440, 445, 451, 455, 458, 472, 477, 479, 493, 495, 504, 511, 515, 517, 522, 524, 528, 532, 536, 544, 568, 580, 592, 611, 612, 620, 622, 628, 632, 633, 645, 658, 659, 662, 670, 675, 686, 694, 697, 699, 702, 703, 706, 709, 710, 711, 712, 714, 716, 723, 743, 746, 750, 751, 755, 757, 761, 768, 770, 772, 773, 776, 778, 780, 782, 783, 789, 790, 792, 796, 801, 807, 813, 815, 828, 829, 832, 833, 834, 835, 836, 837, 838, 840, 841, 845, 847, 863, 875, 878, 879, 883, 888, 893, 895, 904, 905, 914, 922, 925, 928, 930, 932, 938, 940, 942, 943, 947, 949, 952, 960, 967, 969, 973, 974, 976, 980, 981, 984, 985, 986, 987, 991, 997, 998, 1000, 1001, 1003, 1009, 1010, 1011, 1013, 1027, 1031, 1037, 1048, 1049, 1053, 1056, 1058, 1060, 1065, 1066, 1068, 1076, 1080, 1084, 1088, 1092, 1096, 1100, 1102, 1105, 1106, 1107, 1114, 1125, 1129, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1142, 1143, 1144, 1151, 1156, 1167, 1169, 1170, 1172, 1173, 1174, 1175, 1179, 1181, 1182, 1186, 1187, 1188, 1196, 1197, 1200, 1209, 1214, 1218, 1219, 1223, 1226, 1230, 1233, 1238, 1241, 1245, 1246, 1251, 1260, 1262, 1272, 1275, 1277, 1283, 1285, 1286, 1289, 1292, 1293, 1294, 1295, 1297, 1300, 1308, 1310, 1313, 1314, 1315, 1319, 1321, 1327, 1328, 1337, 1338, 1339, 1341, 1345, 1347, 1349, 1350, 1357, 1361, 1367, 1369, 1371, 1374, 1376, 1383, 1386, 1389, 1390, 1391, 1394, 1395, 1396, 1398, 1401, 1404, 1408, 1409, 1412, 1415, 1420, 1421, 1422, 1423, 1426, 1427, 1428, 1429, 1431, 1432, 1433, 1437, 1441, 1442, 1443, 1449, 1462, 1471, 1474, 1480, 1511, 1523, 1537, 1547, 1552, 1561, 1563, 1568, 1587, 1588, 1590, 1592, 1593, 1595, 1598, 1600, 1601, 1617, 1621, 1622, 1626, 1627, 1641, 1644, 1648, 1649, 1651, 1652, 1654, 1656, 1660, 1662, 1668, 1673, 1676, 1687, 1688, 1709, 1723, 1739, 1745, 1746, 1753, 1764, 1766, 1767, 1775, 1781, 1783, 1786, 1790, 1791, 1794, 1797, 1801, 1803, 1814, 1815, 1819, 1820, 1825, 1827, 1835, 1840, 1844, 1846, 1850, 1852, 1853, 1855, 1856, 1859, 1864, 1870, 1871, 1873, 1875, 1881, 1883, 1888, 1893, 1894, 1901, 1922, 1927, 1929, 1943, 1948, 1952, 1980, 1983, 1985, 1988, 1998, 1999]





import statistics
datasets = load_dataset('squad')
g_f1 = []
g_accs = []
g_acce = []
g_sun = []

wrong_ids = []
for i in range(100):
    if i not in incorrect_answer_id:
        text = datasets['train'][i]['context']
        question = datasets['train'][i]['question']
        answers = datasets['train'][i]['answers']
        f1, acc_s, acc_e, sun = cycle_prediction(10, question, text, answers['text'][0])
        if f1[0] != 1:
            wrong_ids.append(i)
            print(wrong_ids)
            analysis(f1, acc_s, acc_e, sun)
        else:
            print("预测正确：", i)
        g_f1.append(f1)
        g_accs.append(acc_s)
        g_acce.append(acc_e)
        g_sun.append(sun)
print(g_f1, g_accs ,g_acce ,g_sun )
temp_initpro = 0
temp_explainpro = 0
explain_score = []
for i in g_sun:
    temp_initpro = temp_initpro + i[0]
    temp_explainpro = temp_explainpro + max(i)
print('模型初始的概率为 :', temp_initpro/len(g_sun), "解释后的概率是:", temp_explainpro/len(g_sun))






