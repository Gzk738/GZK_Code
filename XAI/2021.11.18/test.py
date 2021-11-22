from datasets import load_dataset

datasets = load_dataset('squad', split="validation")
print(datasets)