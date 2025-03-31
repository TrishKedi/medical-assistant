from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# Load dataset
dataset = load_dataset("medalpaca/medical_meadow_medqa")
dataset["train"] = dataset["train"].select(range(2000))
# Format into instruction-based prompt
def format_sample(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset["train"].map(format_sample)
dataset = dataset.train_test_split(test_size=0.05)

# Load tokenizer & model with 4-bit QLoRA config
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config={
        "load_in_4bit":True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }
)

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Tokenize
def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./lora-med-assistant",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=False,
    save_total_limit=2,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()
