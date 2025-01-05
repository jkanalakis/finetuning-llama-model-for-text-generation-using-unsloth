# Fine-Tuning and Text Generation with `unsloth/Llama-3.2-3B-Instruct`

This project demonstrates how to fine-tune a pre-trained language model named `unsloth/Llama-3.2-3B-Instruct` using the Hugging Face `transformers`, `trl`, and `datasets` libraries. The notebook walks through all steps, from installing dependencies to fine-tuning the model and performing text generation. Below, are detailed instructions and explanations of each step.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Steps](#steps)
   - [1. Install Required Libraries](#1-install-required-libraries)
   - [2. Import Dependencies](#2-import-dependencies)
   - [3. Load Pre-Trained Model](#3-load-pre-trained-model)
   - [4. Apply Parameter-Efficient Fine-Tuning (PEFT)](#4-apply-parameter-efficient-fine-tuning-peft)
   - [5. Prepare Dataset and Tokenizer](#5-prepare-dataset-and-tokenizer)
   - [6. Configure the Fine-Tuning Trainer](#6-configure-the-fine-tuning-trainer)
   - [7. Train and Save the Model](#7-train-and-save-the-model)
   - [8. Reload Fine-Tuned Model](#8-reload-fine-tuned-model)
   - [9. Optimize Model for Inference](#9-optimize-model-for-inference)
   - [10. Generate Text](#10-generate-text)
6. [Results](#results)
7. [Customization](#customization)
8. [Contributing](#contributing)
9. [License](#license)

## Overview

This notebook showcases the following:
- Fine-tuning a pre-trained language model for a specific dataset.
- Preparing datasets with conversational templates for instruction tuning.
- Using Parameter-Efficient Fine-Tuning (PEFT) to optimize resource usage.
- Saving, loading, and deploying a fine-tuned model for text generation.

## Features

- **PEFT Fine-Tuning:** Enables resource-efficient training by updating only a subset of model parameters.
- **Custom Dataset Preparation:** Prepares conversational data using ShareGPT templates.
- **Text Generation:** Demonstrates model inference with advanced sampling techniques like top-p sampling and temperature control.
- **Modular Approach:** Code is modular and easily customizable.

## Prerequisites

- Python 3.8 or later
- GPU with CUDA support (recommended for faster training and inference)
- Required Python libraries (see [Setup](#setup))

## Setup

1. Clone this repository or download the notebook.
2. Install the required Python libraries by running the following commands in your terminal:

   ```bash
   pip install unsloth torch transformers datasets trl
   ```

3. Ensure your system has sufficient memory and GPU resources to handle the model.

## Steps

### 1. Install Required Libraries
This step installs the necessary Python libraries:

```python
!pip install unsloth
!pip install torch
!pip install transformers
!pip install datasets
!pip install trl
```

### 2. Import Dependencies
Import essential modules for model handling, dataset preparation, and fine-tuning.

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
```

### 3. Load Pre-Trained Model
Load the `unsloth/Llama-3.2-3B-Instruct` model with 4-bit precision for efficient memory usage.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
```

### 4. Apply Parameter-Efficient Fine-Tuning (PEFT)
Enable PEFT by configuring specific model layers for fine-tuning.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### 5. Prepare Dataset and Tokenizer
Load a dataset, format it with chat templates, and tokenize it.

```python
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(
    lambda examples: {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["conversations"]
        ]
    },
    batched=True
)
```

### 6. Configure the Fine-Tuning Trainer
Initialize the trainer with the model, dataset, and training arguments.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs"
    )
)
```

### 7. Train and Save the Model
Fine-tune the model and save it for later use.

```python
trainer.train()
model.save_pretrained("finetuned_model")
```

### 8. Reload Fine-Tuned Model
Reload the fine-tuned model for inference.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True
)
```

### 9. Optimize Model for Inference
Enable faster inference with model optimizations.

```python
FastLanguageModel.for_inference(model)
```

### 10. Generate Text
Perform text generation using the fine-tuned model.

```python
input_prompt = "Once upon a time"
inputs = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Results

After fine-tuning, the model generates high-quality text that aligns with the dataset's structure and style. The generated text can be used for various NLP tasks like chatbot development, content creation, and more.

## Customization

You can customize:
- **Model:** Replace `unsloth/Llama-3.2-3B-Instruct` with another pre-trained model.
- **Dataset:** Load a different dataset or preprocess it with your templates.
- **Training Parameters:** Adjust learning rate, batch size, and training steps for different results.

## Contributing

Contributions are welcome! Please submit issues or pull requests for improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
