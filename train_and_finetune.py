"""
This code performs the following steps:

1) It sets up a binary classification task using the GPT-2 model on an environmental claims dataset.
2) The initial model is evaluated to establish a baseline.
3) The model is then fine-tuned using LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning method.
4) The fine-tuned model is evaluated and compared to the initial model.
5) Finally, the PEFT (Parameter-Efficient Fine-Tuned) model is saved, reloaded, and evaluated again to ensure consistency.

Throughout the process, the code saves evaluation results and model weights, allowing for later analysis and use of the trained model.
"""

## 0) IMPORTS:
# Import necessary libraries
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import load_dataset
import random
import json

# Set random seed for reproducibility
random.seed(1209)

## 1) LOADING AND EVALUATING A FOUNDATION MODEL:

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Set pad token to be the same as the end of sequence token
tokenizer.pad_token = tokenizer.eos_token

# Load pre-trained GPT-2 model and configure it for binary classification
model = AutoModelForSequenceClassification.from_pretrained('gpt2',
                                                           num_labels=2,
                                                           id2label={0: "no", 1: "yes"},
                                                           label2id={"no": 0, "yes": 1})

# Set output directory for saving results
output_dir = r"C:\Users\Carla\Desktop\Notebooks\modelFineTuinning\results"

# Configure model to recognize padding
model.config.pad_token_id = model.config.eos_token_id

# Load the environmental claims dataset for both train and validation splits
splits = ["train", "validation"]
dataset = {split: load_dataset("climatebert/environmental_claims", split=split) for split in splits}

# Define preprocessing function to tokenize input text
def preprocess_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return tokenized

# Apply preprocessing to both train and validation datasets
encoded_dataset = {split: dataset[split].map(preprocess_function, batched=True) for split in splits}


# Define function to compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary', pos_label=1)
    recall = recall_score(labels, preds, average='binary', pos_label=1)
    f1 = f1_score(labels, preds, average='binary', pos_label=1)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # Sensitivity
        "specificity": specificity,
        "f1": f1
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Evaluate the initial model
eval_result = trainer.evaluate()

# Save initial evaluation results to a JSON file
eval_results_file = f"{output_dir}/evaluation_initial_results.json"
with open(eval_results_file, 'w') as f:
    json.dump(eval_result, f)

## 2) PERFORMING PARAMETER-EFFICIENT FINE-TUNNING:

# Configure LoRA (Low-Rank Adaptation) for fine-tuning
config = LoraConfig(
    r=10,  # Rank
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# Create PEFT (Parameter-Efficient Fine-Tuning) model
peft_model = get_peft_model(model, config)

# Initialize the Trainer with the PEFT model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train the PEFT model
trainer.train()

# Evaluate the fine-tuned PEFT model
eval_result_finetunned = trainer.evaluate()

# Save fine-tuned evaluation results to a JSON file
eval_results_file_finetunned = f"{output_dir}/evaluation_finetunned_results.json"
with open(eval_results_file_finetunned, 'w') as f:
    json.dump(eval_result_finetunned, f)

# Save the PEFT model weights
peft_model.save_pretrained(f'{output_dir}/peft_model')

## 3) PERFORMING INTERFERENCE WITH A PEFT MODEL:

# Load the tokenizer and the PEFT model for inference
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

peft_model = AutoPeftModelForSequenceClassification.from_pretrained('./peft_model',
                                                                    num_labels=2,
                                                                    id2label={0: "no", 1: "yes"},
                                                                    label2id={"no": 0, "yes": 1})
peft_model.config.pad_token_id = peft_model.config.eos_token_id

# Re-setup the Trainer with the loaded PEFT model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Evaluate the loaded PEFT model
peft_eval_result = trainer.evaluate()


## 4) PRINTING FINAL RESULTS:

# Compare the results of initial, fine-tuned, and PEFT models
initial_eval_accuracy = eval_result['eval_accuracy']
finetunned_eval_accuracy = eval_result_finetunned['eval_accuracy']
peft_eval_accuracy = peft_eval_result['eval_accuracy']

print(f"Accuracy of initial model: \n {initial_eval_accuracy}")
print(f"Accuracy of post fine-tuning model:\n {finetunned_eval_accuracy}")
print(f"Accuracy of PEFT model:\n {peft_eval_accuracy}")