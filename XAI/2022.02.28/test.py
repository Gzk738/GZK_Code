from datasets import load_dataset

squad = load_dataset("squad")
train = squad["train"]
validation = squad["validation"]
pass