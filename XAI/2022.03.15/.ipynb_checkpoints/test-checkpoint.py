# from datasets import load_dataset
# squad = load_dataset("squad")
# shuffled_squad = squad.shuffle()
#
# train_contexts = []
# train_questions = []
# train_answers = []
# train_ids = []
#
#
# for i in range(len(shuffled_squad['train'])):
#     train_contexts.append(shuffled_squad['train'][i]['context'])
#     train_questions.append(shuffled_squad['train'][i]['question'])
#     train_answers.append(shuffled_squad['train'][i]['answers'])
#     train_ids.append(shuffled_squad['train'][i]['id'])
import json
from pathlib import Path
with open(r"train_context.txt") as f:
    train_contexts = json.load(f)
with open(r"train_question.txt") as f:
    train_questions = json.load(f)
with open(r"train_answers.txt") as f:
    train_answers = json.load(f)
with open(r"train_ids.txt") as f:
    train_ids = json.load(f)



def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    ids = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    ids.append(qa['id'])

    return contexts, questions, answers, ids

validation_contexts, validation_questions, validation_answers, validation_ids = read_squad(r'dev-v2.0.json')




import nltk as tk
import re

sep_train_contexts = []
sep_train_questions = []
sep_train_answers = []
error_index = []
null_answer = {'text': '[NULL]', 'answer_start': 0}
for i in range(len(train_contexts)):
    tokens = tk.sent_tokenize(train_contexts[i])
    for token in tokens:
        if train_answers[i]['text'][0] in token:
            try:
                answer_start = re.search(train_answers[i]['text'][0], token)
                answer = {'text': train_answers[i]['text'], 'answer_start': answer_start.span()[0]}
                sep_train_contexts.append(token)

                sep_train_answers.append(answer)
                sep_train_questions.append(train_questions[i])
            except:
                error_index.append(i)

                # print(i)
        # else:
        #     sep_train_contexts.append('[NULL]' + token)
        #     sep_train_answers.append(null_answer)
        #     sep_train_questions.append(train_questions[i])
print("error_index : ", len(error_index))

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text'][0]
        if type(answer['answer_start']) == list:
            temp = answer['answer_start'][0]
            answer['answer_start'] = temp
            start_idx = answer['answer_start']
        else:
            start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters
        else:
            answer['answer_end'] = end_idx

add_end_idx(sep_train_answers, sep_train_contexts)

add_end_idx(train_answers, train_contexts)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

sep_train_encodings = tokenizer(sep_train_contexts, sep_train_questions, truncation=True, padding=True)

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    print("len len(answers) : ", len(answers))
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))

        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


add_token_positions(sep_train_encodings, sep_train_answers)

add_token_positions(train_encodings, train_answers)

import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

sep_train_dataset = SquadDataset(sep_train_encodings)

train_dataset = SquadDataset(train_encodings)

from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

import datetime
import json
import argparse
import collections

import numpy as np
import os
import re
import string
import sys

ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id
print(datetime.datetime.now())


def predict(inputs):
    output = model(inputs)
    return output.start_logits, output.end_logits


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
                    [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)


def predict_qt(question, text):
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id,
                                                                cls_token_id)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    ground_truth = '13'

    start_scores, end_scores = predict(input_ids)

    return (' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]))


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans
def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()
def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))
def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    # print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        exact = 100.0 * sum(exact_scores.values()) / total
        f1 = 100.0 * sum(f1_scores.values()) / total

        return collections.OrderedDict([
            ('exact', exact),
            ('f1', f1),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        # print('exact------->', 100.0 * sum(exact_scores.values()) / total)
        # print('f1------->', 100.0 * sum(f1_scores.values()) / total)
        # print('total------->', total)
        return collections.OrderedDict([
          ('exact', 100.0 * sum(exact_scores.values()) / total),
          ('f1', 100.0 * sum(f1_scores.values()) / total),
          ('total', total),
        ])

def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
      main_eval['%s_%s' % (prefix, k)] = new_eval[k]
def test_valisdation():
    va_evl_dict = {}

    for i in range(1500):
        question = validation_questions[i]
        text = validation_contexts[i]
        if len(text) <= 512:
            answer = predict_qt(question, text)
            va_evl_dict[str(validation_ids[i])] = answer
    time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    json_dict = json.dumps(va_evl_dict)
    filname = time + "validation_answers.txt"

    fo = open("./results/" + filname, "w", encoding='utf-8')
    fo.write(json_dict)
    fo.close()

    with open(r"dev-v2.0.json") as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']

    with open(r"./results/" + filname) as f:
        preds = json.load(f)

    na_probs = {k: 0.0 for k in preds}
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,1.0)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,1.0)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')

        print(json.dumps(out_eval, indent=2))
    return json.dumps(out_eval, indent=2)

from torch.utils.data import DataLoader
from transformers import AdamW
import datetime

print (datetime.datetime.now())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)
list_loss = []
acc = []
for epoch in range(20):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
    acc.append(test_valisdation())
    list_loss.append(float(loss))
    model_name = "epoch" + str(epoch) + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.pth'
    torch.save(model, './models/' + model_name)

model.eval()

print (datetime.datetime.now())
print("loss : ", list_loss)
print("acc : ", acc)

train_loader = DataLoader(sep_train_dataset, batch_size=16, shuffle=True)

for epoch in range(20):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
    acc.append(test_valisdation())
    list_loss.append(float(loss))
    model_name = "epoch" + str(epoch) + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.pth'
    torch.save(model, './models/' + model_name)
print (datetime.datetime.now())
print("loss : ", list_loss)
print("acc : ", acc)