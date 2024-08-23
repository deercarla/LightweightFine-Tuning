# Fine-Tuning GPT-2 with LoRA for Environmental Claims Classification

This project demonstrates the fine-tuning of the GPT-2 model using the Low-Rank Adaptation (LoRA) technique. The task involves binary classification of environmental claims. We compare the performance of the original pre-trained model and the fine-tuned model using various metrics.

## Project Overview

1. **Loading and Evaluating a Foundation Model**
2. **Performing Parameter-Efficient Fine-Tuning (PEFT)**
3. **Inference with the Fine-Tuned Model**
4. **Results and Conclusions**

## 1. Loading and Evaluating a Foundation Model

### Model

- **Model:** GPT-2
- **Tokenizer:** GPT2Tokenizer
- **Pre-trained Model:** `AutoModelForSequenceClassification`
- **Task:** Binary classification

We use GPT-2, a widely adopted language model, known for its versatility and capability to generate coherent text. It has been pre-trained on diverse internet text, making it a strong foundation model. The choice of GPT-2 is influenced by its relatively small size and compatibility with sequence classification tasks, making it suitable for fine-tuning with limited computational resources.

### Dataset

- **Dataset:** [Environmental Claims Dataset](https://huggingface.co/datasets/climatebert/environmental_claims)
- **Splits:** Train and validation
- **Source:** Hugging Face Datasets library

The Environmental Claims dataset is a well-suited choice for this task as it provides real-world examples of textual claims related to environmental issues. This makes it relevant for tasks like misinformation detection or environmental awareness. The dataset's availability in the Hugging Face library ensures easy integration and handling, and its modest size makes it manageable for fine-tuning in resource-constrained environments.

### Initial Evaluation

We evaluate the base model's performance on the validation split before fine-tuning. The dataset is tokenized using the GPT-2 tokenizer, with padding and truncation applied to a maximum length of 128 tokens.

### Evaluation Metrics

We use the following metrics to evaluate model performance:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score

### Initial Results

```json
{
  "eval_loss": 0.700258195400238,
  "eval_accuracy": 0.298,
  "eval_precision": 0.0,
  "eval_recall": 0.0,
  "eval_specificity": 0.0,
  "eval_f1": 0.0,
  "eval_runtime": 489.2829,
  "eval_samples_per_second": 2.044,
  "eval_steps_per_second": 0.255
}
```

The initial evaluation indicates that the pre-trained GPT-2 model performs poorly on this specific task without fine-tuning, with an accuracy of 29.8%.

## 2. Performing Parameter-Efficient Fine-Tuning (PEFT)

### LoRA Configuration

- **Rank (r):** 10
- **Alpha:** 32
- **Target Modules:** ['c_attn', 'c_proj']
- **Dropout:** 0.1

LoRA introduces low-rank adaptations to certain layers of the model, allowing efficient fine-tuning with fewer trainable parameters. This reduces the computational cost and memory footprint compared to full fine-tuning.

### Training

The fine-tuning process involves training the model for three epochs with a learning rate of 5e-5. We use the same training and validation datasets. The decision to use LoRA was driven by its ability to fine-tune large models with limited resources, an essential consideration given the project constraints.

## 3. Inference with the Fine-Tuned Model

After fine-tuning, we save the PEFT model and reload it for evaluation. We ensure consistency in evaluation by using the same metrics as in the initial evaluation.

### Fine-Tuned Model Evaluation

```json
{
  "eval_loss": 0.0,
  "eval_accuracy": 1.0,
  "eval_precision": 1.0,
  "eval_recall": 1.0,
  "eval_specificity": 1.0,
  "eval_f1": 1.0,
  "eval_runtime": 423.0169,
  "eval_samples_per_second": 2.364,
  "eval_steps_per_second": 0.295
}
```

The fine-tuned model achieved perfect scores across all evaluation metrics. However, such results suggest overfitting, as the model may not generalize well to unseen data. The reduced runtime compared to the initial evaluation also indicates improved efficiency in model execution.

## 4. Results and Conclusions

### Comparison of Models

#### Initial Model
- **Accuracy:** 0.298
- **Precision:** 0.0
- **Recall:** 0.0
- **Specificity:** 0.0
- **F1 Score:** 0.0
- **Evaluation Time:** 489.2829 seconds

#### Fine-Tuned Model (LoRA)
- **Accuracy:** 1.0
- **Precision:** 1.0
- **Recall:** 1.0
- **Specificity:** 1.0
- **F1 Score:** 1.0
- **Evaluation Time:** 423.0169 seconds

### Analysis

The fine-tuned model's perfect accuracy indicates it learned the dataset's patterns exceptionally well, but it also raises concerns about overfitting. The original model's poor performance highlighted the need for fine-tuning to adapt the model to this specific binary classification task.

Using LoRA provided significant advantages, including reduced memory usage and faster training times, allowing for efficient fine-tuning even with limited computational resources. However, a disadvantage is the potential for underfitting if the low-rank adaptations do not capture sufficient task-specific features. Moreover, LoRA's application is limited to specific layers and tasks, which can constrain its generalizability across different models and tasks.

### Conclusion

In this project, we successfully fine-tuned the GPT-2 model using LoRA for the binary classification of environmental claims. While the fine-tuned model achieved high accuracy, the possibility of overfitting suggests the need for further validation on a separate test set or with additional regularization techniques. Overall, LoRA proved to be an efficient and effective fine-tuning method for adapting large pre-trained models to specific tasks with limited computational resources.