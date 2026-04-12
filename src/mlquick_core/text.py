from __future__ import annotations

import re

import jieba
import pandas as pd

try:
    from nltk.corpus import stopwords

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


jieba.setLogLevel(jieba.logging.INFO)

_BASIC_ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "this", "that", "these", "those", "i",
    "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
}

_BASIC_CHINESE_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这",
}


def detect_language(text: str) -> str:
    if pd.isna(text) or text == "":
        return "unknown"

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", str(text)))
    total_chars = len(re.sub(r"\s+", "", str(text)))
    if total_chars == 0:
        return "unknown"
    return "chinese" if chinese_chars / total_chars > 0.3 else "english"


def preprocess_text_column(
    series: pd.Series,
    language: str = "auto",
    remove_stopwords: bool = True,
    min_word_length: int = 2,
) -> pd.Series:
    processed_texts: list[str] = []

    for raw_text in series.fillna(""):
        text = str(raw_text).strip()
        if not text:
            processed_texts.append("")
            continue

        detected_language = detect_language(text) if language == "auto" else language
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if detected_language == "chinese":
            words = jieba.lcut(text)
            if remove_stopwords:
                words = [
                    word for word in words
                    if word not in _BASIC_CHINESE_STOPWORDS and len(word) >= min_word_length
                ]
            processed_texts.append(" ".join(words))
            continue

        words = text.lower().split()
        if remove_stopwords:
            if NLTK_AVAILABLE:
                try:
                    english_stopwords = set(stopwords.words("english"))
                except LookupError:
                    english_stopwords = _BASIC_ENGLISH_STOPWORDS
            else:
                english_stopwords = _BASIC_ENGLISH_STOPWORDS
            words = [
                word for word in words
                if word not in english_stopwords and len(word) >= min_word_length
            ]
        processed_texts.append(" ".join(words))

    return pd.Series(processed_texts, index=series.index)
