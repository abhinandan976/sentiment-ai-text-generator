# AI Sentiment Text Generator (Flask)

## Overview
This project is a Flask-based AI Text Generator that:

- Detects the sentiment of user-provided prompts (positive, negative, neutral) using a **custom-trained DistilBERT model**.
- Generates coherent paragraphs aligned with the detected sentiment using a **GPT-2 style text generation model**.
- Allows optional manual sentiment selection and adjustable text length for more control.
- Provides an interactive web interface for entering prompts and viewing generated text.

The project demonstrates **end-to-end AI integration**: sentiment analysis, text generation, and web deployment.

---

## Dataset
- **Dataset used**: [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- This dataset contains tweets labeled with sentiment (positive, negative, neutral) and was used to train the custom sentiment model.

---

## Sentiment Model
- **Model**: `DistilBERT` fine-tuned for sentiment analysis.
- **Saved as**: `ai_sentiment_analyzer(DistillBERT).joblib`
- **Purpose**: Classifies user input text into `positive`, `negative`, or `neutral`.
- **Training process**:
  - Preprocessed tweets from the Twitter Sentiment Analysis Dataset.
  - Used Hugging Face Transformers and PyTorch.
  - Fine-tuned DistilBERT for classification.
  - Saved the trained model using `joblib` for fast loading in the Flask app.

---

## Text Generation Model
- **Model**: `distilgpt2` from Hugging Face Transformers.
- Generates text aligned with the detected sentiment.
- A sentiment-guiding prefix is added to the prompt to control tone:
  - Positive → “Write an optimistic, uplifting paragraph about:”
  - Negative → “Write a critical, cautionary paragraph about:”
  - Neutral → “Write a neutral, objective paragraph about:”

---

## Project Structure
ai_text_generator/
│── app.py # Flask backend
│── model.py # Sentiment detection & text generation logic
│── ai_sentiment_analyzer(DistillBERT).joblib # Saved sentiment model
│── requirements.txt
│── README.md
├── templates/
│ └── index.html # HTML UI
└── static/
└── style.css # Optional styling

---

## Setup Instructions (Local)
1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
