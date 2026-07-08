"""Flask web interface for AI Quiz Generator — Team MechMind."""

from __future__ import annotations

import os
import tempfile

from flask import Flask, jsonify, render_template, request

from quiz_generator import evaluate_metrics, generate_quiz
from quiz_generator.pipeline import acquire_input

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/generate")
def generate():
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    max_per_type = int(payload.get("max_per_type", 3))
    seed = int(payload.get("seed", 42))

    if not text:
        return jsonify({"error": "Please provide study text."}), 400
    if len(text) < 50:
        return jsonify({"error": "Text is too short. Provide at least 50 characters."}), 400

    items = generate_quiz(text, max_per_type=max_per_type, seed=seed)
    metrics = evaluate_metrics(items, text)
    return jsonify({"quiz": [i.to_dict() for i in items], "metrics": metrics})


@app.post("/generate-from-pdf")
def generate_from_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No PDF file uploaded."}), 400

    pdf = request.files["file"]
    if not pdf.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    max_per_type = int(request.form.get("max_per_type", 3))
    seed = int(request.form.get("seed", 42))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = acquire_input(pdf_path=tmp_path)
        if len(text) < 50:
            return jsonify({"error": "Could not extract enough text from PDF."}), 400
        items = generate_quiz(text, max_per_type=max_per_type, seed=seed)
        metrics = evaluate_metrics(items, text)
        return jsonify({"quiz": [i.to_dict() for i in items], "metrics": metrics, "extracted_chars": len(text)})
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
