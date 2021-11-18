# from datasets import load_dataset
#
# dataset = load_dataset("qangaroo", 'medhop', split="validation")
# row = dataset[300]
# print(dataset[0])

#
# import json
# import json
# import re
# import sys
#
#
# sys.setrecursionlimit(1000000)
#
#
# def deal_json_invaild(text):
#     if type(text) != str:
#         raise Exception("参数接受的是字符串类型")
#     # text = re.search(r"\{.*\}", text).group()
#     text = re.sub(r"\n|\t|\r|\r\n|\n\r|\x08|\\", "", text)
#     try:
#         json.loads(text)
#     except json.decoder.JSONDecodeError as err:
#         temp_pos = int(re.search(r"\(char (\d+)\)", str(err)).group(1))
#         temp_list = list(text)
#         while True:
#             if temp_list[temp_pos] == "\"" or "}":
#                 if temp_list[temp_pos - 1] == "{":
#                     break
#                 elif temp_list[temp_pos - 1] == (":" or "{") and temp_list[temp_pos - 2] == ("\"" or ":" or "["):
#                     break
#                 elif temp_list[temp_pos] == "|\n|\t|\r|\r\n|\n\r| ":
#                     temp_list[temp_pos] = re.sub(temp_list[temp_pos], "", temp_list[temp_pos])
#                     text = "".join(temp_list)
#                 elif temp_list[temp_pos] == "\"":
#                     temp_list[temp_pos] = re.sub(temp_list[temp_pos], "“", temp_list[temp_pos])
#                     text = "".join(temp_list)
#                 elif temp_list[temp_pos] == "}":
#                     temp_list[temp_pos - 1] = re.sub(temp_list[temp_pos], "\"", temp_list[temp_pos])
#                     text = "".join(temp_list)
#                     temp_pos -= 1
#             temp_pos -= 1
#         return deal_json_invaild(text)
#     else:
#         return text
# #读取文件
# file_path = r"D:\software\hotpot_dev_distractor_v1.txt"
# with open(file_path, 'r') as fp:
#     #a = deal_json_invaild(fp.read())
#     js_data = json.loads(str(fp.read()))
#     print(type(js_data))
# # 遍历键值
# for key in js_data:
#     print(key)
#
#
#


from transformers import AutoModelWithHeads

model = AutoModelWithHeads.from_pretrained("bert-base-uncased")
adapter_name = model.load_adapter("AdapterHub/bert-base-uncased-pf-hotpotqa", source="hf")
model.active_adapters = adapter_name

model