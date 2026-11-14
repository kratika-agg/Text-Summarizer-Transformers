# Text Summarization using T5 Transformers

An NLP and Generative AI project focused on generating concise summaries of conversations using the T5 (Text-to-Text Transfer Transformer) architecture.

## Overview

This project explores abstractive text summarization using a pretrained T5-small model.

The system takes a dialogue as input and generates a concise summary based on the provided conversational context.

The project uses **T5 for Conditional Generation**, where the generated output (summary) is conditioned on the input dialogue.

## Tech Stack

- Python
- PyTorch
- Pandas
- Hugging Face Transformers
- T5 (Text-to-Text Transfer Transformer)
- Jupyter Notebook

## Dataset

The project uses the [SAMSum Dataset](https://huggingface.co/datasets/Samsung/samsum), which contains conversational dialogues paired with human-written summaries.

The dataset is divided into:
- Training dataset
- Validation dataset
- Test dataset

For the current implementation, a reproducible sample was selected:
- Training samples: 4,000
- Validation samples: 500
- Random state: 42

## Implementation

### 1. Data Preprocessing

- Removed unnecessary newline characters.
- Normalized whitespace.
- Removed HTML tags.
- Converted dialogue and summary text to lowercase.

### 2. Tokenization

Used the pretrained T5 tokenizer to convert dialogues and summaries into token IDs.

Configuration:
- Maximum input sequence length: 512
- Maximum target sequence length: 150
- Padding and truncation enabled.

### 3. Model Selection

Used the pretrained `t5-small` checkpoint with `T5ForConditionalGeneration`.

T5 follows an encoder-decoder architecture and supports conditional text generation, making it suitable for dialogue summarization.

The smaller model variant was selected to keep computational requirements manageable.

### 4. Model Training

The model was fine-tuned using Hugging Face's `Trainer` API.

**Training Configuration**

| Parameter | Value |
|---|---|
| Pretrained model | t5-small |
| Epochs | 6 |
| Weight decay | 0.01 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Warmup steps | 500 |
| Evaluation strategy | Every epoch |
| Checkpoint saving | Every epoch |
| Output directory | `./results` |

The training process automatically saves model checkpoints in the configured output directory.

### 5. Training Results

- Training completed for 6 epochs.
- Total training steps: 3,000.
- Final training loss: approximately 0.904.

Training and validation metrics are monitored across epochs to observe model learning and generalization.

## Project Structure

```text
Transformer/
│
├── text_summarizer.ipynb
├── samsum-train.csv
├── samsum-validation.csv
├── samsum-test.csv
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Transformer
```

### 2. Install dependencies

```bash
pip install "transformers[torch]" pandas
```

### 3. Prepare the dataset

Place the SAMSum CSV files in the project root directory:

- `samsum-train.csv`
- `samsum-validation.csv`
- `samsum-test.csv`

### 4. Launch JupyterLab

```bash
jupyter lab
```

Open `text_summarizer.ipynb` and execute the notebook cells.

## Current Progress

- [x] Initial project setup
- [x] Library imports and environment verification
- [x] SAMSum dataset loading
- [x] Random sampling (4,000 training and 500 validation)
- [x] Data cleaning and preprocessing
- [x] T5-small tokenizer initialization
- [x] Training and validation data tokenization
- [x] T5 model fine-tuning
- [x] Saving the model and tokenizer
- [x] Reloading the saved model and tokenizer
- [ ] Model evaluation
- [ ] Summary generation and inference
- [ ] Application integration


## Project Status

**In Progress**

The initial model fine-tuning stage has been completed. Further work will focus on evaluating the model and generating summaries from unseen dialogues.