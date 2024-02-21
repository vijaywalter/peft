from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import evaluate
from tqdm import tqdm

# Constants and Configuration
MODEL_ID = "google/flan-t5-xxl"
DATASET_NAME = "samsum"
OUTPUT_DIR = "lora-flan-t5-xxl"
PEFT_MODEL_ID = "results"

# Load dataset
dataset = load_dataset(DATASET_NAME)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def compute_max_length(sequences):
    lengths = [len(x) for x in sequences]
    return int(np.percentile(lengths, 85)), int(np.percentile(lengths, 90))

def preprocess_dataset(dataset, tokenizer):
    # Tokenize and prepare inputs and targets
    tokenized_datasets = {}
    for split in ['train', 'test']:
        tokenized_datasets[split] = dataset[split].map(
            lambda x: {"input_ids": tokenizer(x["dialogue"], truncation=True)["input_ids"],
                       "labels": tokenizer(x["summary"], truncation=True)["input_ids"]},
            batched=True,
            remove_columns=["dialogue", "summary", "id"]
        )
    return tokenized_datasets

tokenized_datasets = preprocess_dataset(dataset, tokenizer)

# Compute max lengths for source and target sequences
max_source_length, max_target_length = compute_max_length(tokenized_datasets["train"]["input_ids"] + tokenized_datasets["test"]["input_ids"])

def preprocess_function(examples):
    # Prepare model inputs and labels with proper padding and truncation
    model_inputs = tokenizer(examples["input_ids"], max_length=max_source_length, truncation=True, padding="max_length")
    labels = tokenizer(examples["labels"], max_length=max_target_length, truncation=True, padding="max_length")
    
    # Replace padding token id in labels with -100
    labels["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100) for label in lab] for lab in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Preprocess datasets
for split in tokenized_datasets.keys():
    tokenized_datasets[split] = tokenized_datasets[split].map(preprocess_function, batched=True)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Train
trainer.train()

# Save trained model and tokenizer
trainer.model.save_pretrained(PEFT_MODEL_ID)
tokenizer.save_pretrained(PEFT_MODEL_ID)

# Evaluation
def evaluate_model(test_dataset, model, tokenizer, metric):
    predictions, references = [], []
    for sample in tqdm(test_dataset):
        input_ids = tokenizer(sample["input_ids"], return_tensors="pt", truncation=True).input_ids
        outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference = tokenizer.decode(sample["labels"], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(reference)
    
    return metric.compute(predictions=predictions, references=references)

# Load metric
metric = evaluate.load("rouge")

# Evaluate the model
results = evaluate_model(tokenized_datasets["test"], model, tokenizer, metric)
print(f"ROUGE scores: {results}")

