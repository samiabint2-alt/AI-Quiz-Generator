"""
AI Quiz Generator pipeline (Team MechMind methodology).

Steps:
  1. Input acquisition
  2. Text preprocessing
  3. Content analysis
  4. Question generation
  5. Answer extraction
  6. Question classification
  7. Quiz structuring
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        if pkg.startswith("punkt"):
            nltk.data.find(f"tokenizers/{pkg}")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))

_nlp = None


def _get_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        _nlp = False
    return _nlp


@dataclass
class QuizItem:
    qtype: str  # MCQ | TrueFalse | ShortAnswer
    question: str
    options: Optional[List[str]] = None
    correct: str = ""
    source_sentence: str = ""

    def to_dict(self) -> Dict:
        d = {
            "type": self.qtype,
            "question": self.question,
            "correct": self.correct,
            "source_sentence": self.source_sentence,
        }
        if self.options is not None:
            d["options"] = self.options
        return d


def extract_text_from_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def acquire_input(text: Optional[str] = None, pdf_path: Optional[str] = None) -> str:
    """Step 1: Input acquisition."""
    if pdf_path:
        return extract_text_from_pdf(pdf_path)
    if text and text.strip():
        return text.strip()
    raise ValueError("Provide text or a PDF path.")


def preprocess(raw: str):
    """Step 2: Text preprocessing."""
    text = raw.replace("\r\n", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) >= 5]
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    tokens_nostop = [w for w in tokens if w not in STOP and len(w) > 2]
    return text, sentences, tokens_nostop


def analyze_content(sentences: List[str], tokens_nostop: List[str]) -> Dict:
    """Step 3: Content analysis."""
    fd = FreqDist(tokens_nostop)
    keywords = [w for w, _ in fd.most_common(25)]
    entities: List[str] = []
    nlp = _get_spacy()
    if nlp:
        joined = " ".join(sentences[:80])
        doc = nlp(joined[:500000])
        entities = list({e.text for e in doc.ents if len(e.text) > 2})[:20]
    return {"keywords": keywords, "entities": entities}


def pick_distractors(answer: str, pool: List[str], k: int = 3) -> List[str]:
    ans_l = answer.lower()
    cand = [p for p in pool if p.lower() != ans_l and len(p) > 2]
    random.shuffle(cand)
    return cand[:k]


def sentence_to_short_answer(sentence: str, keyword: str) -> Optional[QuizItem]:
    if keyword.lower() not in sentence.lower():
        return None
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    blanked = pattern.sub("_______", sentence, count=1)
    if "_______" not in blanked:
        return None
    return QuizItem(
        "ShortAnswer",
        f"What word or phrase completes this sentence?\n{blanked}",
        None,
        keyword,
        sentence,
    )


def sentence_to_true_false(sentence: str) -> Optional[QuizItem]:
    s = sentence.strip()
    if len(s.split()) < 6:
        return None
    neg = re.search(r"\b(not|never|no|without|isn't|aren't|wasn't|weren't)\b", s, re.I)
    stmt = s[:-1] if s.endswith(".") else s
    correct = "False" if neg else "True"
    return QuizItem("TrueFalse", f"True or False: {stmt}.", None, correct, sentence)


def sentence_to_mcq(sentence: str, keywords: List[str]) -> Optional[QuizItem]:
    words = sentence.split()
    if len(words) < 6:
        return None
    content = [
        w
        for w in words
        if re.match(r"^[A-Za-z][a-zA-Z-]*$", w) and w.lower() not in STOP
    ]
    if not content:
        return None
    answer = max(content, key=len)
    if len(answer) < 4:
        return None
    qtext = sentence.replace(answer, "________", 1)
    if "________" not in qtext:
        return None
    distractors = pick_distractors(answer, keywords + content, k=3)
    while len(distractors) < 3:
        distractors.append("distractor_" + random.choice("abcdef"))
    opts = distractors[:3] + [answer]
    random.shuffle(opts)
    return QuizItem(
        "MCQ",
        "Multiple choice — what fits the blank?\n" + qtext,
        opts,
        answer,
        sentence,
    )


def generate_questions(sentences: List[str], analysis: Dict, max_per_type: int) -> List[QuizItem]:
    """Steps 4–6: Question generation, answer extraction, classification."""
    kws = analysis["keywords"]
    items: List[QuizItem] = []
    used_s = set()

    for s in sentences:
        if sum(1 for i in items if i.qtype == "MCQ") >= max_per_type:
            break
        qi = sentence_to_mcq(s, kws)
        if qi and s not in used_s:
            items.append(qi)
            used_s.add(s)

    for s in sentences:
        if sum(1 for i in items if i.qtype == "TrueFalse") >= max_per_type:
            break
        if s in used_s:
            continue
        qi = sentence_to_true_false(s)
        if qi:
            items.append(qi)
            used_s.add(s)

    for kw in kws[:15]:
        if sum(1 for i in items if i.qtype == "ShortAnswer") >= max_per_type:
            break
        for s in sentences:
            if s in used_s:
                continue
            qi = sentence_to_short_answer(s, kw)
            if qi:
                items.append(qi)
                used_s.add(s)
                break

    return items


def structure_quiz(items: List[QuizItem]) -> List[Dict]:
    """Step 7: Quiz structuring."""
    return [i.to_dict() for i in items]


def generate_quiz(
    text: str,
    max_per_type: int = 4,
    seed: int = 42,
) -> List[QuizItem]:
    """Full pipeline: preprocess → analyze → generate → structure."""
    random.seed(seed)
    _, sentences, tokens_nostop = preprocess(text)
    if not sentences:
        return []
    analysis = analyze_content(sentences, tokens_nostop)
    return generate_questions(sentences, analysis, max_per_type)


def quiz_to_json(items: List[QuizItem]) -> str:
    return json.dumps(structure_quiz(items), indent=2, ensure_ascii=False)
