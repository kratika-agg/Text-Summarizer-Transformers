# Text Summarization Using Transformers

**An End-to-End NLP Project | T5 | Hugging Face | PyTorch | FastAPI**

An NLP project focused on building an intelligent dialogue summarization system using Transformer-based sequence-to-sequence models. The system aims to transform lengthy conversations into concise, meaningful summaries while preserving their essential information.

The project uses the **SAMSum conversational dataset** and fine-tunes the pretrained **T5-small model** using Hugging Face Transformers.

---

## 1. Project Overview

Conversations often contain redundant information, multiple speakers, and lengthy exchanges that make it difficult to identify the most important information quickly.

This project aims to address that challenge by developing a dialogue summarization pipeline that:

- Processes raw conversational text.
- Cleans and preprocesses dialogue data.
- Converts text into model-compatible token representations.
- Fine-tunes a pretrained T5 Transformer model.
- Generates concise summaries using beam search.
- Provides a reusable summarization function for inference.
- Presents an initial user interface for text summarization.
- Plans to integrate the trained model with a FastAPI backend.

The project follows a progressive development approach, starting with model development and moving towards application integration and deployment.

## 2. Technology Stack 

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| NLP Framework | Hugging Face Transformers |
| Pretrained Model | T5-small |
| Tokenization | T5Tokenizer |
| Data Processing | Pandas |
| Model Training | Hugging Face Trainer API |
| Development Environment | Jupyter Notebook, VS Code |
| Frontend | HTML, CSS |
| Backend (Planned) | FastAPI |
| Hardware Acceleration | Apple Metal Performance Shaders (MPS), where supported |

## 3. Dataset

The project uses the **SAMSum dataset**, which consists of conversational dialogues paired with human-written summaries.

Each record contains:

- `id`: Unique identifier for the conversation.
- `dialogue`: Original conversational text.
- `summary`: Human-written reference summary.

### Dataset Files

- `samsum-train.csv`
- `samsum-validation.csv`
- `samsum-test.csv`

### Initial Dataset Sampling

For the initial experimentation phase, a smaller subset was selected to make model development and training more manageable.

| Dataset | Original Records | Selected Records |
|---|---:|---:|
| Training | 14,732 | 4,000 |
| Validation | 818 | 500 |

Random sampling was performed using `random_state=42` to ensure reproducibility.

The complete dataset is reserved for future experimentation and performance comparison.

## 4. Data Preprocessing

Raw dialogue and summary text undergo preprocessing before being passed to the tokenizer.

The implemented cleaning operations include:

1. Removal of newline characters.
2. Removal of HTML tags.
3. Removal of unnecessary whitespace.
4. Removal of leading and trailing spaces.
5. Conversion of text to lowercase.

These operations help maintain consistent text formatting and prepare the dataset for tokenization.

## 5. Tokenization

The pretrained `t5-small` tokenizer is used to convert dialogues and reference summaries into numerical token representations.

### Tokenization Configuration

| Parameter | Dialogue Inputs | Summary Labels |
|---|---:|---:|
| Maximum Length | 512 | 150 |
| Padding | `max_length` | `max_length` |
| Truncation | Enabled | Enabled |

The tokenized dialogue inputs are used as model inputs, while the tokenized reference summaries are prepared as labels for supervised fine-tuning.

## 6. Model Architecture and Fine-Tuning

The project uses **T5 (Text-to-Text Transfer Transformer)**, a sequence-to-sequence architecture that treats NLP tasks as text-to-text transformations.

The pretrained `t5-small` checkpoint is loaded using Hugging Face Transformers.

### Model Implementation

- Model class: `T5ForConditionalGeneration`
- Tokenizer class: `T5Tokenizer`
- Training interface: Hugging Face `Trainer`
- Task: Abstractive dialogue summarization

The model is fine-tuned on the sampled SAMSum training dataset, with the validation dataset used during training.

## 7. Model Saving and Reloading

After fine-tuning, both the model and tokenizer are saved locally using Hugging Face's `save_pretrained()` method.

```python
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")
```

The saved components are subsequently reloaded using `from_pretrained()`.

```python
model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model"
)
```

This ensures that the saved model can be reused for inference without repeating the training process.

## 8. Core Summarization Pipeline

A reusable `summarize_dialogue()` function has been implemented to generate summaries from new conversational inputs.

The function follows three primary stages:

**1. Input Preprocessing**

The incoming dialogue is cleaned using the existing preprocessing function.

**2. Tokenization**

The cleaned dialogue is converted into input IDs and attention masks using the trained tokenizer.

**3. Summary Generation and Decoding**

The model generates summary token IDs using its `generate()` method. The tokenizer then decodes those token IDs into readable natural language text.

### Generation Configuration

| Parameter | Value |
|---|---:|
| Maximum Generation Length | 150 |
| Number of Beams | 4 |
| Early Stopping | Enabled |

Beam search is used to explore multiple candidate sequences during summary generation.

### Initial Inference Testing

The summarization function has been tested using a custom dialogue discussing:

- Growth of artificial intelligence across industries.
- Adoption of machine learning systems.
- Advances in deep learning.
- Bias and fairness in AI.
- AI regulations and accountability.
- Explainability and responsible AI development.

The fine-tuned model successfully generated a concise summary capturing the central ideas of the conversation.

This confirms that the initial inference pipeline is functional. Systematic evaluation of summary quality remains a future task.

## 9. Frontend Development

The initial frontend interface has been developed using **HTML and CSS**.

The objective is to provide a simple, intuitive interface through which users can enter conversational text and view the generated summary.

### 9.1 User Interface Components

The initial interface includes:

- **Application Heading:** Displays the Text Summarizer application title.
- **Model Identification:** Introduces the application as a Hugging Face Transformer-based summarization system.
- **Text Input Area:** Provides a textarea for entering or pasting dialogue.
- **Submit Button:** Provides the interface control for submitting text for summarization.
- **Summary Output Section:** A dedicated area designed to display the generated summary.

### 9.2 Styling and Layout

The interface uses embedded CSS within `index.html`.

The implemented styling includes:

- Arial typography for readability.
- A light background (`#FFF7ED`) and white content container.
- A centered layout with a maximum container width of 600px.
- Rounded corners and subtle box shadows.
- Consistent spacing and padding.
- Styled textarea and submit button.
- Button hover effects.
- A highlighted summary output section.

### 9.3 Frontend Project Files

| File | Description |
|---|---|
| `index.html` | Contains the HTML structure and CSS styling of the application. |
| `app.py` | Reserved for FastAPI backend development and model integration. |

**Current implementation status:** The initial frontend structure and styling are complete. Backend functionality, API communication and dynamic summary generation through the interface are still under development.

## 10. Development Roadmap

The project is being developed in stages, following the accompanying course implementation.

### Phase 1 — Dataset Preparation and Model Development

- [x] Project setup and environment configuration
- [x] SAMSum dataset loading
- [x] Random sampling of training and validation data
- [x] Text cleaning and preprocessing
- [x] T5-small tokenizer initialization
- [x] Training and validation data tokenization
- [x] Transformer model initialization
- [x] Model fine-tuning
- [x] Model and tokenizer saving
- [x] Reloading the saved model and tokenizer
- [x] Core summarization function implementation
- [x] Custom dialogue inference testing

### Phase 2 — Frontend Development

- [x] HTML structure for the summarization interface
- [x] CSS styling and layout
- [x] User input area for conversational text
- [x] Submit button
- [x] Summary output section

### Phase 3 — Backend Development with FastAPI

*Upcoming*

- [ ] FastAPI environment setup
- [ ] Backend application development
- [ ] Loading the saved model and tokenizer in the backend
- [ ] Creating summarization API endpoints
- [ ] Processing user requests
- [ ] Returning generated summaries to the frontend

### Phase 4 — Client-Server Integration

*Upcoming*

- [ ] Understanding the client-server architecture
- [ ] Connecting the frontend with the backend
- [ ] Implementing JavaScript logic
- [ ] Connecting frontend components with FastAPI endpoints
- [ ] Handling user input and API responses
- [ ] Displaying generated summaries dynamically

### Phase 5 — Final Application and Demonstration

*Upcoming*

- [ ] End-to-end application testing
- [ ] Functional verification
- [ ] User interface improvements
- [ ] Final project demonstration

## 11. Project Structure and Components

The repository is organised into model development and application development components.

```text
Text-Summarizer-Transformers/
│
├── TextSummarizerApp/
│   ├── app.py
│   ├── index.html
│   └── saved_summary_model/    # Local model artifacts
│
├── text_summarizer.ipynb
│
├── samsum-train.csv
├── samsum-validation.csv
├── samsum-test.csv
│
├── .gitignore
└── README.md
```

The `text_summarizer.ipynb` notebook contains the model development and experimentation workflow, while `TextSummarizerApp/` contains the application frontend and the planned backend implementation.

**Note:** The trained model directory is maintained locally and excluded from Git tracking because model weights can be large. Dataset and other project files are managed separately.

## 12. Getting Started

### Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL

cd Text-Summarizer-Transformers
```

### Create a Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install "transformers[torch]" pandas sentencepiece jupyter
```

### Prepare the Dataset

Ensure the SAMSum CSV files are available in the project directory before running the notebook.

### Launch JupyterLab

```bash
jupyter lab
```

Open `text_summarizer.ipynb` and execute the notebook cells in sequence.

The saved model directory is generated after model fine-tuning and is used by the application during the backend integration stage.

## 13. Future Improvements

Potential areas for further experimentation include:

- Training on the complete SAMSum dataset.
- Evaluating generated summaries using ROUGE and other suitable NLP evaluation metrics.
- Comparing different training configurations.
- Improving summary quality and inference performance.
- Completing the FastAPI backend.
- Integrating the frontend with the summarization API.
- Enhancing the user interface.
- Performing end-to-end application testing.

## 14. Project Status

**Status: In Progress**

The initial model development, fine-tuning, model persistence, core dialogue summarization pipeline and frontend interface have been implemented.

The next development stage focuses on building the FastAPI backend, loading the saved model for inference, and integrating the frontend with the backend to create a functional text summarization application.

---

**Built as part of an ongoing exploration of Natural Language Processing, Transformer architectures and practical AI application development.**
