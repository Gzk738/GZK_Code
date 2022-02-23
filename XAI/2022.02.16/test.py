import json
with open(r"D:\software\github\GZK_Code\XAI\2022.02.16\answers copy.txt",  "r")as f:
    json_strs = f.readlines()
#删除所有的空行
for json_str in json_strs:
    if json_str == '\n':
        json_strs.remove(json_str)

data = []
for i in json_strs:
    data.append(json.loads(i))




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




"""
原模型准确率
"""
acc = 0
temp = 0
for i in range(4141):
    f1 = 0
    f1 = compute_f1(data[i]["answer"], data[i]["preds"][0])
    acc += f1
    temp +=1
print("原模型准确率", acc/temp)

"""只根据atribution得准确率"""
bestanswer_ids = []
for i in data:
    bestanswer_ids.append(i["attributions"].index(max(i["attributions"])))

acc = 0
temp = 0
for i in range(4141):
    f1 = 0
    f1 = compute_f1(data[i]["answer"], data[i]["preds"][bestanswer_ids[i]])
    acc += f1
    temp +=1
print("只根据atribution得准确率", acc/temp)


"""只根据atribution+attentions准确率"""
bestanswer_ids = []
aa_acore = []
for i in data:
    aa_acore = [i["attributions"][aa] + i["start_score"][aa] + i["end_score"][aa] for aa in range(len(i["attributions"]))]
    bestanswer_ids.append(aa_acore.index(max(aa_acore)))

acc = 0
temp = 0
for i in range(4141):
    f1 = 0
    f1 = compute_f1(data[i]["answer"], data[i]["preds"][bestanswer_ids[i]])
    acc += f1
    temp +=1
print("只根据atribution+attentions准确率", acc/temp)
"""最好的准确率"""
acc = 0
temp = 0
best_f1_index = []
for i in range(4141):
    f1 = []
    #f1 = compute_f1(data[i]["answer"], data[i]["preds"][0])
    for pred in data[i]["preds"]:
        f1.append(compute_f1(data[i]["answer"], pred))
    acc += max(f1)
    temp +=1
    best_f1_index.append(f1.index(max(f1)))
print("最好的准确率", acc/temp)
import numpy as np
import matplotlib.pyplot as plt
xs =  np.linspace(0,2, 11)
ys =  np.linspace(0,2, 11)
zs =  np.linspace(0,2, 11)
thresholds = []
Accurates = []
for x in range(len(xs)):
    for y in range(len(ys)):
        for z in range(len(zs)):
            temp0 = xs[x]
            temp1 = ys[y]
            temp2 = zs[z]
            thresholds.append([temp0, temp1, temp2])

def computer_ths_acc(threshold):
    if threshold != [0.0,0.0,0.0]:
        bestanswer_ids = []
        aa_acore = []
        for i in data:
            aa_acore = [i["attributions"][aa]*threshold[0] + i["start_score"][aa]*threshold[1] + i["end_score"][aa]*threshold[2] for aa in
                        range(len(i["attributions"]))]
            bestanswer_ids.append(aa_acore.index(max(aa_acore)))

        acc = 0
        temp = 0
        for i in range(4141):
            f1 = 0
            f1 = compute_f1(data[i]["answer"], data[i]["preds"][bestanswer_ids[i]])
            acc += f1
            temp += 1
        #print("只根据atribution+attentions准确率", acc / temp)
        Accurates.append(acc / temp)






"""不同权重的atribution+attentions准确率"""
for threshold in range(len(thresholds)):
    #print(thresholds[threshold])
    computer_ths_acc(thresholds[threshold])
    print(thresholds[threshold])



plt.plot(range(len(thresholds) -1), Accurates)
plt.show()