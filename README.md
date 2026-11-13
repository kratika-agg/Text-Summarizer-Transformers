# Text Summarization using Transformers

An ongoing NLP project focused on exploring text summarization using Transformer-based models.

## Project Overview

This project aims to build a text summarization system using the SAMSum dialogue dataset and Hugging Face Transformers.

The objective is to generate concise summaries of conversations while preserving their key information.

## Tech Stack

- Python
- PyTorch
- Pandas
- Hugging Face Transformers
- Jupyter Notebook

## Dataset

The project uses the SAMSum dataset, which contains conversational dialogues paired with human-written summaries.

Dataset files:
- `samsum-train.csv`
- `samsum-validation.csv`
- `samsum-test.csv`

## Project Structure

```text
Transformer/
├── text_summarizer.ipynb
├── samsum-train.csv
├── samsum-validation.csv
├── samsum-test.csv
├── .gitignore
└── README.md
```

## Current Progress

- [x] Initial project setup
- [x] Library imports and environment checks
- [ ] Data exploration and preprocessing
- [ ] Tokenization
- [ ] Transformer model implementation
- [ ] Model training
- [ ] Evaluation and inference

## Getting Started

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Text-Summarizer-Transformers
```

Install the required libraries:

```bash
pip install torch pandas transformers
```

Launch JupyterLab:

```bash
jupyter lab
```

Open `text_summarizer.ipynb` to explore the implementation.

## Project Status

**In Progress** — This repository will be updated as development progresses.