# Text Summarization using Transformers

An NLP project focused on building a dialogue summarization system using Transformer-based models and Hugging Face.

The project uses the **SAMSum dataset** to generate concise summaries of conversations while preserving their important information.

## Project Overview

The goal is to fine-tune a T5 Transformer model for dialogue summarization.

Development is being carried out incrementally, beginning with data exploration, preprocessing and tokenization before moving to model training and evaluation.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Pandas
- Jupyter Notebook
- T5 (Text-to-Text Transfer Transformer)

## Dataset

The project uses the **SAMSum dataset**, which contains conversational dialogues paired with human-written summaries.

Dataset files:

- `samsum-train.csv`
- `samsum-validation.csv`
- `samsum-test.csv`

### Initial Dataset Sampling

For the initial development and experimentation phase, a smaller subset has been selected to make preprocessing and training experiments faster.

| Dataset | Original size | Selected size |
|---|---:|---:|
| Training | 14,732 | 4,000 |
| Validation | 818 | 500 |

Random sampling is performed using `random_state=42` to ensure reproducibility.

The complete dataset can be used in subsequent experiments.

## Data Preprocessing

The following cleaning operations have been applied to dialogue and summary text:

- Removal of newline characters.
- Removal of extra whitespace.
- Removal of HTML tags.
- Removal of leading and trailing spaces.
- Conversion of text to lowercase.

## Tokenization

The project uses the pretrained `t5-small` tokenizer from Hugging Face.

Current tokenization configuration:

| Parameter | Dialogue | Summary |
|---|---:|---:|
| Maximum sequence length | 512 | 150 |
| Padding | max_length | max_length |

Tokenized dialogue inputs and summary labels are prepared for the model fine-tuning stage.

## Project Structure

```text
Text-Summarizer-Transformers/
├── text_summarizer.ipynb
├── .gitignore
└── README.md
```

The SAMSum CSV files are used locally and are not intended to be committed to GitHub.

## Current Progress

- [x] Initial project setup
- [x] Library imports and environment verification
- [x] SAMSum dataset loading
- [x] Random sampling of training and validation data
- [x] Text cleaning and preprocessing
- [x] T5-small tokenizer initialization
- [x] Training and validation data tokenization
- [ ] Transformer model implementation
- [ ] Model training and fine-tuning
- [ ] Model evaluation
- [ ] Summary generation and inference
- [ ] Application integration

## Development Environment

The project is currently being developed locally using:

- macOS (Apple Silicon)
- Jupyter Notebook
- PyTorch with Apple Metal Performance Shaders (MPS), where supported

## Getting Started

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Text-Summarizer-Transformers
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install "transformers[torch]" pandas
```

Launch JupyterLab:

```bash
jupyter lab
```

Open `text_summarizer.ipynb` to explore the implementation.

Ensure the SAMSum CSV files are available in the project directory before running the notebook.

## Project Status

**In Progress** — Initial data preparation and tokenization have been completed. The next stage will focus on implementing and fine-tuning the T5 model.