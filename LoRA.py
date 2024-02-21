from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
import torch
from peft import PeftModel, PeftConfig

# Define constants
model_id = "google/flan-t5-xxl"
peft_model_id = "results"
output_dir = "lora-flan-t5-xxl"
max_source_length = 128  # Adjust based on your needs
max_target_length = 140  # Adjust based on your needs
label_pad_token_id = -100

# Load dataset
dataset = load_dataset("samsum")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

# Preprocess function
def preprocess_function(sample, padding="max_length"):
    inputs = ["summarize: " + item for item in sample["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])

# Define LoRA configuration and prepare model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# Create trainer and train model
trainer = Seq2SeqTrainer(model=model, args=training_args, data_collator=data_collator, train_dataset=tokenized_dataset["train"])
model.config.use_cache = False
trainer.train()

# Save model and tokenizer
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

# Load peft model and tokenizer
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
model.eval()

# Inference function
def predict(sample):
    input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_
