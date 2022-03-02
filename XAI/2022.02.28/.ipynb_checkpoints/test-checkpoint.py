from datasets import load_dataset

squad = load_dataset("squad")

sep_train = squad["train"][:10]
sep_validation = squad["validation"][:10]
sep_dataset = {}
sep_dataset["train"] = sep_train
sep_dataset["validation"] = sep_validation

from datasets import DatasetDict
dataset = Dataset.from_dict(sep_train)



pass