# app.py
from flask import Flask, render_template, request, redirect, url_for
from model import get_sentiment_label, generate_sentiment_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    detected_sentiment = ""
    manual_sentiment = ""
    length = 120
    result_text = ""
    error = None

    if request.method == "POST":
        try:
            prompt = request.form.get("prompt", "").strip()
            # optional manual sentiment selector (user override)
            manual_sentiment = request.form.get("manual_sentiment", "").strip().lower()
            length_str = request.form.get("length", "120").strip()
            try:
                length = max(20, min(400, int(length_str)))
            except ValueError:
                length = 120

            if not prompt:
                error = "Please enter a prompt."
            else:
                detected_sentiment = get_sentiment_label(prompt)
                chosen_sentiment = manual_sentiment if manual_sentiment in ("positive","negative","neutral") else detected_sentiment
                result_text = generate_sentiment_text(prompt, sentiment=chosen_sentiment, max_new_tokens=length)
        except Exception as e:
            error = f"Error during generation: {e}"

    return render_template("index.html",
                           prompt=prompt,
                           detected_sentiment=detected_sentiment,
                           manual_sentiment=manual_sentiment,
                           length=length,
                           result_text=result_text,
                           error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
