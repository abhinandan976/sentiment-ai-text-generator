# model.py
from transformers import pipeline, set_seed
import threading

# We'll load models once (thread-safe)
_model_lock = threading.Lock()
_sentiment_pipeline = None
_generation_pipeline = None

def _load_pipelines():
    global _sentiment_pipeline, _generation_pipeline
    with _model_lock:
        if _sentiment_pipeline is None:
            # Sentiment: DistilBERT fine-tuned on SST-2
            _sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        if _generation_pipeline is None:
            # Text generation: lightweight GPT2 variant
            _generation_pipeline = pipeline("text-generation", model="distilgpt2")
            set_seed(42)

def get_sentiment_label(text, neutral_threshold=0.60):
    """
    Returns one of: 'positive', 'negative', 'neutral'
    We use pipeline's label + score and interpret low confidence as neutral.
    """
    _load_pipelines()
    if not text or not text.strip():
        return "neutral"
    res = _sentiment_pipeline(text[:512])[0]  # limit length
    label = res.get("label", "").lower()      # 'POSITIVE' or 'NEGATIVE'
    score = float(res.get("score", 0.0))
    # If confidence is low, classify as neutral
    if score < neutral_threshold:
        return "neutral"
    if label.startswith("pos"):
        return "positive"
    elif label.startswith("neg"):
        return "negative"
    else:
        return "neutral"

def generate_sentiment_text(prompt, sentiment="neutral", max_new_tokens=120):
    """
    Generate a paragraph influenced by sentiment.
    - prompt: user prompt (string)
    - sentiment: 'positive' | 'negative' | 'neutral'
    - max_new_tokens: target length (approx tokens)
    """
    _load_pipelines()
    # Prepare a sentiment-guiding prefix to control tone
    prefix_map = {
        "positive": "Write an optimistic, uplifting paragraph about: ",
        "negative": "Write a critical, cautionary paragraph about: ",
        "neutral" : "Write a neutral, objective paragraph about: "
    }
    prefix = prefix_map.get(sentiment, prefix_map["neutral"])
    gen_input = prefix + prompt.strip()

    # transformers' pipeline supports max_length; we approximate:
    # to be compatible across versions we use max_length = len(prompt_tokens)+max_new_tokens approximated.
    # Simpler: set max_length to 200-300 which works for short paragraphs.
    try:
        # prefer modern arg max_new_tokens - but pipeline may not accept it depending on library version
        output = _generation_pipeline(gen_input, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=True, temperature=0.8)
        text = output[0]["generated_text"]
    except TypeError:
        # fallback if max_new_tokens not supported
        output = _generation_pipeline(gen_input, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.8)
        text = output[0]["generated_text"]

    # Remove the guiding prefix from generated text if it repeated; find first occurrence of prompt and return from there.
    # Try to extract generated paragraph about the prompt.
    # If generator repeats the prefix, remove it:
    if gen_input.lower() in text.lower():
        # If generator included our full gen_input at start, drop until after it
        idx = text.lower().find(gen_input.lower())
        text = text[idx + len(gen_input):].strip()
    else:
        # If not, attempt to drop prefix words
        for p in prefix_map.values():
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()
                break

    # Ensure it is a short paragraph — trim to first two sentences if too long
    # This is a naive split on periods for brevity; we keep up to 3 sentences.
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) > 3:
        text = '. '.join(sentences[:3]) + '.'
    else:
        # return as-is but ensure trailing period
        if not text.endswith('.'):
            text = text.strip()
            if not text.endswith('.'):
                text = text + '.'
    # final cleanup
    return text.strip()
