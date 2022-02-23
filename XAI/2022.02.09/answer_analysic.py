import ast
import json
with open(r"answers.txt")as f:
    lines = f.readlines()
#删除所有的空行
for line in lines:
    if line == '\n' or len(line) == 0:
        lines.remove(line)
#删除最后得回车换行符
lines = [line[:len(line) - 1] for line in lines]
print(lines[:6])
ids = []
truths = []
answers = []
atributions = []
foeword_atten = []
backword_atten = []

for i in range(len(lines)):
    try:
        if i % 6 == 0 and i != 0:
            ids.append(lines[i - 6])
            truths.append(lines[i - 5])
            answers.append(ast.literal_eval(lines[i - 4]))
            atributions.append(lines[i - 3])
            foeword_atten.append(lines[i - 2])
            backword_atten.append(lines[i - 1])
    except:
        pass

temp = []
for i in range(965):
    temp.append(json.loads(foeword_atten[i]))
foeword_atten = temp

temp = []
for i in range(965):
    temp.append(json.loads(backword_atten[i]))
backword_atten = temp

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


best_answersids = []
acc = 0
temp = 0
str_tensor = []
for i in atributions[0:965]:
    value = i.split(",")[::3]
    value = [float(i[8:]) for i in value]
    str_tensor.append((value))



for onetest_answers in str_tensor:
    best_answersids.append(onetest_answers.index(max(onetest_answers)))

all_attribution = []
for i in range(965):
    onetest_all_contribution = []
    for j in range(len(str_tensor[i])):
        onetest_all_contribution.append(str_tensor[i][j] + backword_atten[i][j] + foeword_atten[i][j])
    all_attribution.append(onetest_all_contribution.index(max(onetest_all_contribution)))

for i in range((965)):
    f1 = 0
    f1 = compute_f1(answers[i][all_attribution[i]], truths[i])
    acc += f1
    temp +=1
    print(answers[i][best_answersids[i]])
    print(temp,"--",f1, "*"*144)
    print(truths[i])

print(acc/temp)