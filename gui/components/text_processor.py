#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理组件 - 处理中英文文本预处理和特征提取
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import List, Optional, Tuple

# 导入文本处理库
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
    # 抑制jieba的日志输出
    jieba.setLogLevel(jieba.logging.INFO)
except ImportError:
    JIEBA_AVAILABLE = False
    warnings.warn("jieba未安装，中文文本处理功能将不可用")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn未安装，文本特征提取功能将不可用")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK未安装，英文文本处理功能将不可用")

class TextProcessor:
    """文本处理器"""

    def __init__(self):
        self.stopwords_cache = {}
        self._init_stopwords()

    def _init_stopwords(self):
        """初始化停用词"""
        # 中文停用词
        self.chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还是', '为', '但是', '可以', '这个', '那个',
            '什么', '怎么', '这样', '那样', '因为', '所以', '如果', '虽然', '可是', '然而'
        }

        # 英文停用词
        self.english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

    def detect_language(self, text: str) -> str:
        """
        检测文本语言

        Args:
            text: 输入文本

        Returns:
            语言类型 ('chinese', 'english', 'unknown')
        """
        if pd.isna(text) or text == "":
            return "unknown"

        # 计算中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', str(text)))
        total_chars = len(re.sub(r'\s+', '', str(text)))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        return "chinese" if chinese_ratio > 0.3 else "english"

    def preprocess_text_column(self, series: pd.Series,
                             language: str = "auto",
                             remove_stopwords: bool = True,
                             min_word_length: int = 2) -> pd.Series:
        """
        预处理文本列

        Args:
            series: pandas Series，包含文本数据
            language: 语言类型 ("auto", "chinese", "english")
            remove_stopwords: 是否移除停用词
            min_word_length: 最小词长度

        Returns:
            预处理后的文本Series
        """
        processed_texts = []

        for text in series:
            if pd.isna(text) or text == "":
                processed_texts.append("")
                continue

            text = str(text).strip()

            # 自动检测语言
            if language == "auto":
                detected_lang = self.detect_language(text)
            else:
                detected_lang = language

            # 清理文本
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)  # 保留中英文和数字
            text = re.sub(r'\s+', ' ', text)  # 合并多个空格

            if detected_lang == "chinese":
                processed_text = self._process_chinese_text(
                    text, remove_stopwords, min_word_length
                )
            else:
                processed_text = self._process_english_text(
                    text, remove_stopwords, min_word_length
                )

            processed_texts.append(processed_text)

        return pd.Series(processed_texts)

    def _process_chinese_text(self, text: str, remove_stopwords: bool,
                            min_word_length: int) -> str:
        """处理中文文本"""
        if not JIEBA_AVAILABLE:
            warnings.warn("jieba未安装，无法进行中文分词")
            return text

        # 中文分词处理
        words = jieba.lcut(text)

        # 移除停用词和短词
        if remove_stopwords:
            words = [word for word in words
                    if word not in self.chinese_stopwords
                    and len(word) >= min_word_length
                    and not word.isspace()]

        processed_text = ' '.join(words)
        return processed_text

    def _process_english_text(self, text: str, remove_stopwords: bool,
                            min_word_length: int) -> str:
        """处理英文文本"""
        # 转换为小写
        text = text.lower()
        words = text.split()

        # 移除停用词和短词
        if remove_stopwords:
            # 尝试使用nltk停用词
            stopwords_set = self._get_english_stopwords()
            words = [word for word in words
                    if word not in stopwords_set
                    and len(word) >= min_word_length
                    and word.isalpha()]

        processed_text = ' '.join(words)
        return processed_text

    def _get_english_stopwords(self) -> set:
        """获取英文停用词"""
        if NLTK_AVAILABLE:
            try:
                return set(stopwords.words('english'))
            except:
                # 如果nltk数据未下载，使用基础停用词
                return self.english_stopwords
        else:
            return self.english_stopwords

    def extract_features(self, text_data: pd.Series,
                        max_features: int = 1000,
                        method: str = "tfidf") -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        从文本数据提取特征

        Args:
            text_data: 预处理后的文本数据
            max_features: 最大特征数
            method: 特征提取方法 ("tfidf" 或 "count")

        Returns:
            (特征矩阵, 特征名称)
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn未安装，无法进行特征提取")
            return None, None

        # 过滤空文本
        non_empty_texts = text_data.dropna()
        non_empty_texts = non_empty_texts[non_empty_texts.str.strip() != '']

        if len(non_empty_texts) == 0:
            warnings.warn("没有有效的文本数据用于特征提取")
            return None, None

        try:
            if method == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),  # 1-gram和2-gram
                    min_df=2,  # 至少出现在2个文档中
                    max_df=0.8,  # 最多出现在80%的文档中
                    token_pattern=r'(?u)\b\w+\b'  # 支持中文
                )
            else:
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    token_pattern=r'(?u)\b\w+\b'
                )

            features = vectorizer.fit_transform(non_empty_texts)
            feature_names = vectorizer.get_feature_names_out()

            return features, list(feature_names)

        except Exception as e:
            warnings.warn(f"文本特征提取失败: {str(e)}")
            return None, None

    def get_text_statistics(self, text_data: pd.Series) -> dict:
        """
        获取文本统计信息

        Args:
            text_data: 文本数据

        Returns:
            统计信息字典
        """
        if text_data.empty:
            return {}

        non_empty_texts = text_data.dropna()
        non_empty_texts = non_empty_texts[non_empty_texts.str.strip() != '']

        if len(non_empty_texts) == 0:
            return {
                'total_texts': len(text_data),
                'non_empty_texts': 0,
                'empty_texts': len(text_data),
                'language_distribution': {}
            }

        # 语言分布
        language_counts = non_empty_texts.apply(self.detect_language).value_counts()
        language_distribution = language_counts.to_dict()

        # 文本长度统计
        text_lengths = non_empty_texts.str.len()
        word_counts = non_empty_texts.str.split().str.len()

        stats = {
            'total_texts': len(text_data),
            'non_empty_texts': len(non_empty_texts),
            'empty_texts': len(text_data) - len(non_empty_texts),
            'language_distribution': language_distribution,
            'avg_text_length': text_lengths.mean(),
            'max_text_length': text_lengths.max(),
            'min_text_length': text_lengths.min(),
            'avg_word_count': word_counts.mean(),
            'max_word_count': word_counts.max(),
            'min_word_count': word_counts.min()
        }

        return stats

    def create_vocabulary_data(self, text_data: pd.Series,
                             top_n: int = 20) -> dict:
        """
        创建词汇表数据用于可视化

        Args:
            text_data: 文本数据
            top_n: 返回前N个高频词

        Returns:
            词汇表数据
        """
        if text_data.empty:
            return {}

        # 预处理文本
        processed_text = self.preprocess_text_column(text_data)

        # 合并所有文本
        all_text = ' '.join(processed_text.dropna().astype(str))

        if not all_text.strip():
            return {}

        # 分词统计
        words = all_text.split()
        word_freq = {}

        for word in words:
            if len(word) >= 2:  # 过滤单字符
                word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 返回前N个高频词
        return {
            'words': [word for word, freq in sorted_words[:top_n]],
            'frequencies': [freq for word, freq in sorted_words[:top_n]],
            'total_words': len(word_freq),
            'total_tokens': len(words)
        }