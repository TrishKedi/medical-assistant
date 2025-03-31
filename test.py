from datasets import load_dataset

dataset = load_dataset("medalpaca/medical_meadow_medqa")
print(dataset["train"][0])
