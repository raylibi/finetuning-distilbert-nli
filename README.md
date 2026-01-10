# Fine-Tuning DistilBERT for Natural Language Inference (MNLI)

## 🎓 Student Identification

**Group Information:**
- **Group Members**: 
  - Sahrul Ridho Firdaus - 1103223009
  - [Member 2 Name - NIM]

**Course Information:**
- **Course**: Deep Learning
- **Assignment**: Final Term - Task 1 (MNLI)
- **Date**: January 2026

## 📋 Project Overview

This repository contains a comprehensive implementation of fine-tuning the **DistilBERT** model for **Multi-Genre Natural Language Inference (MNLI)** task. The project demonstrates the application of transfer learning techniques to classify textual entailment relationships between premise-hypothesis pairs into three categories: **Entailment**, **Neutral**, and **Contradiction**.

## 🎯 Purpose

The primary purpose of this repository is to:
- Demonstrate practical implementation of transformer-based models for NLI tasks
- Fine-tune a pre-trained DistilBERT model on the GLUE MNLI dataset
- Evaluate model performance using standard metrics
- Provide a working example of Natural Language Inference classification
- Serve as an educational resource for deep learning and NLP enthusiasts

## 🚀 Project Description

This project implements a **Natural Language Inference (NLI)** system using the DistilBERT architecture. The model is trained on the MNLI (Multi-Genre Natural Language Inference) dataset from the GLUE benchmark to classify the relationship between two sentences.

### Key Features:
- **Model Architecture**: DistilBERT-base-uncased (distilled version of BERT)
- **Task**: Multi-class classification (3 labels)
- **Dataset**: GLUE MNLI (Multi-Genre Natural Language Inference)
- **Framework**: HuggingFace Transformers, PyTorch
- **Training**: GPU-accelerated with mixed precision (FP16)

### Classification Labels:
1. **Entailment (0)**: The hypothesis logically follows from the premise
2. **Neutral (1)**: The hypothesis might be true given the premise but doesn't necessarily follow
3. **Contradiction (2)**: The hypothesis contradicts the premise

## 📊 Models and Performance Metrics

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | distilbert-base-uncased |
| Number of Labels | 3 (Entailment, Neutral, Contradiction) |
| Batch Size | 64 |
| Training Epochs | 3 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW (weight decay: 0.01) |
| Precision | FP16 (Mixed Precision) |

### Performance Metrics

The model is evaluated using the following metrics:
- **Accuracy**: Primary metric for overall classification performance
- **Evaluation Strategy**: Performed at the end of each epoch
- **Validation Set**: MNLI validation_matched dataset

### Training Configuration

```
- Output Directory: ./mnli_model
- Evaluation Strategy: Epoch-based
- Save Strategy: Epoch-based
- Best Model Loading: Enabled
- Save Total Limit: 2 checkpoints
- GPU Acceleration: Enabled with FP16 precision
```

## 📁 Repository Structure

```
finetuning-distilbert-nli/
│
├── finetuning-distilbert-nli.ipynb    # Main Jupyter notebook with complete implementation
├── README.md                           # Project documentation (this file)
├── requirements.txt                    # Python dependencies
├── mnli_model/                         # Training checkpoints (generated during training)
└── final_mnli/                         # Final trained model (generated after training)
```

## 🧭 How to Navigate This Repository

### Step-by-Step Workflow:

1. **Setup Environment**
   - Install required dependencies from `requirements.txt`
   - Verify GPU availability for faster training

2. **Data Loading**
   - Load the GLUE MNLI dataset using HuggingFace datasets library
   - Explore the dataset structure and label distribution

3. **Data Preprocessing**
   - Tokenize premise-hypothesis pairs using DistilBERT tokenizer
   - Apply truncation and padding for uniform sequence length

4. **Model Training**
   - Initialize DistilBERT model with 3 output labels
   - Configure training arguments and Trainer
   - Fine-tune the model on MNLI training set
   - Evaluate on validation_matched dataset

5. **Model Testing**
   - Load the trained model
   - Test on custom premise-hypothesis pairs
   - Analyze predictions and confidence scores

### Main Notebook Sections:

- **SETUP**: Install required libraries
- **Import**: Import necessary modules and configure settings
- **Load Data & Check Output**: Load MNLI dataset and explore structure
- **Tokenizer**: Preprocess text data using DistilBERT tokenizer
- **Model & Training**: Define model, metrics, and training configuration
- **Testing**: Evaluate model on custom examples

## 🛠️ Installation and Usage

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM

### Installation Steps

**Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

```python
# Load the trained model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./final_mnli")
model.eval()

# Make predictions
premise = "A soccer player is running across the field."
hypothesis = "A person is playing sports."

inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    pred_label = model.config.id2label[pred_id]

print(f"Prediction: {pred_label}")
```

## 📦 Requirements

```txt
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
accelerate>=0.20.0
scikit-learn>=1.2.0
torch>=2.0.0
numpy>=1.24.0
```

**Note**: See `requirements.txt` for complete list of dependencies.


## 📝 Results and Findings

The fine-tuned DistilBERT model successfully learns to classify Natural Language Inference relationships. Example predictions:

| Premise | Hypothesis | Prediction |
|---------|-----------|-----------|
| A soccer player is running across the field. | A person is playing sports. | **ENTAILMENT** |
| A man is inspecting the uniform of a figure. | The man is sleeping on the couch. | **CONTRADICTION** |
| The wedding party took pictures inside the building. | The photos were the best ever taken. | **NEUTRAL** |

## 🔬 Technical Insights

- **Model Size**: DistilBERT is 40% smaller and 60% faster than BERT-base
- **Training Time**: Approximately 30-45 minutes per epoch on modern GPU (V100/A100)
- **Inference Speed**: Real-time predictions with sub-second latency
- **Memory Usage**: ~2GB GPU memory during inference

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements. Suggestions and feedback are welcome!

---

**Last Updated**: January 2026  
