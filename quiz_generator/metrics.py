"""Performance metrics from project slides: relevance, accuracy proxy, readability."""

from __future__ import annotations

from typing import Dict, List

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from .pipeline import QuizItem, STOP


def evaluate_metrics(items: List[QuizItem], original: str) -> Dict:
    orig = set(word_tokenize(original.lower())) - STOP
    rel_scores = []
    read_scores = []
    valid = 0

    for it in items:
        if not it.question or not it.correct:
            continue
        valid += 1
        qtok = set(word_tokenize(it.question.lower())) - STOP
        overlap = len(orig & qtok) / max(1, len(qtok))
        rel_scores.append(overlap)
        words = word_tokenize(it.question)
        awl = sum(len(w) for w in words) / max(1, len(words))
        read_scores.append(min(1.0, 6.0 / max(3.0, awl)))

    count = len(items)
    return {
        "count": count,
        "valid_questions": valid,
        "relevance_mean": round(sum(rel_scores) / max(1, len(rel_scores)), 3),
        "readability_mean": round(sum(read_scores) / max(1, len(read_scores)), 3),
        "accuracy_proxy": round(valid / max(1, count), 3),
    }
