"""
Text Processing Pipeline con spaCy Integration

Pipeline NLP per text analysis, feature extraction,
e embedding generation usando spaCy e Ollama.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import spacy
from spacy.lang.en import English
from spacy.tokens import Doc

from ..ollama.client import OllamaClient
from ..ollama.models import EmbeddingRequest
from .models import MemoryEntry, ContentType, ContentFormat
from .exceptions import ProcessingError, EmbeddingError

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Async text processor con spaCy e Ollama integration.

    Provides text analysis, feature extraction,
    e embedding generation per memory entries.
    """

    def __init__(self, ollama_client: OllamaClient, model_name: str = "en_core_web_sm"):
        """
        Initialize text processor.

        Args:
            ollama_client: Configured Ollama client per embeddings
            model_name: spaCy model name to load
        """
        self.ollama_client = ollama_client
        self.model_name = model_name
        self.nlp: Optional[English] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load spaCy model con error handling."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")

            # Optimize pipeline per performance
            if "tagger" not in self.nlp.pipe_names:
                logger.warning("POS tagger not available in model")
            if "ner" not in self.nlp.pipe_names:
                logger.warning("NER not available in model")

        except OSError as e:
            logger.error(f"Failed to load spaCy model {self.model_name}: {e}")
            logger.info("Falling back to blank English model")
            self.nlp = English()

    async def process_text(self, text: str, include_embedding: bool = True) -> Dict[str, Any]:
        """
        Process text con full NLP pipeline.

        Args:
            text: Text to process
            include_embedding: Whether to generate embedding

        Returns:
            Processing results con features e embedding

        Raises:
            ProcessingError: Se il processing fallisce
        """
        try:
            start_time = time.time()

            # Run spaCy processing in thread pool per async compatibility
            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, self._process_with_spacy, text)

            # Extract features
            features = self._extract_features(doc)

            # Generate embedding se richiesto
            embedding = None
            if include_embedding:
                embedding = await self._generate_embedding(text)
                features["embedding"] = embedding

            processing_time = (time.time() - start_time) * 1000
            features["processing_time_ms"] = processing_time

            logger.debug(f"Processed text ({len(text)} chars) in {processing_time:.2f}ms")
            return features

        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            raise ProcessingError(f"Text processing failed: {e}") from e

    def _process_with_spacy(self, text: str) -> Doc:
        """Process text con spaCy (sync operation)."""
        if self.nlp is None:
            raise ProcessingError("spaCy model not loaded")

        # Limit text length per performance
        max_length = 1000000  # 1M chars
        if len(text) > max_length:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_length}")
            text = text[:max_length]

        return self.nlp(text)

    def _extract_features(self, doc: Doc) -> Dict[str, Any]:
        """Extract linguistic features from spaCy Doc."""
        # Keywords (lemmatized non-stop words)
        keywords = []
        for token in doc:
            if (not token.is_stop and
                not token.is_punct and
                not token.is_space and
                len(token.lemma_) > 2 and
                token.pos_ in ["NOUN", "ADJ", "VERB", "PROPN"]):
                keywords.append(token.lemma_.lower())

        # Remove duplicates preserving order
        keywords = list(dict.fromkeys(keywords))[:50]  # Limit to 50 keywords

        # Named entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Sentence segmentation
        sentences = [sent.text.strip() for sent in doc.sents]

        # Text statistics
        stats = {
            "char_count": len(doc.text),
            "word_count": len([token for token in doc if not token.is_space]),
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        }

        # Complexity score (heuristic based on various factors)
        complexity_score = self._calculate_complexity(doc, stats)

        # Basic sentiment (placeholder - could integrate sentiment model)
        sentiment = self._calculate_sentiment(doc)

        return {
            "keywords": keywords,
            "entities": entities,
            "sentences": sentences,
            "stats": stats,
            "complexity_score": complexity_score,
            "sentiment": sentiment,
            "pos_tags": [(token.text, token.pos_) for token in doc[:20]],  # First 20 tokens
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks][:10]  # First 10 chunks
        }

    def _calculate_complexity(self, doc: Doc, stats: Dict) -> int:
        """Calculate content complexity score (1-10)."""
        factors = []

        # Average sentence length
        avg_sent_len = stats.get("avg_sentence_length", 0)
        if avg_sent_len > 20:
            factors.append(3)
        elif avg_sent_len > 15:
            factors.append(2)
        else:
            factors.append(1)

        # Vocabulary diversity (TTR - Type-Token Ratio)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        if tokens:
            ttr = len(set(tokens)) / len(tokens)
            if ttr > 0.7:
                factors.append(3)
            elif ttr > 0.5:
                factors.append(2)
            else:
                factors.append(1)

        # Named entity density
        ent_density = len(doc.ents) / len(doc) if len(doc) > 0 else 0
        if ent_density > 0.1:
            factors.append(2)
        else:
            factors.append(1)

        # POS tag diversity
        pos_tags = set(token.pos_ for token in doc)
        if len(pos_tags) > 8:
            factors.append(2)
        else:
            factors.append(1)

        # Calculate final score
        base_score = sum(factors)
        return min(max(base_score, 1), 10)  # Clamp to 1-10

    def _calculate_sentiment(self, doc: Doc) -> float:
        """Calculate basic sentiment score (-1 to 1)."""
        # Placeholder implementation - could integrate TextBlob or spaCy sentiment
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "perfect", "success"}
        negative_words = {"bad", "terrible", "awful", "horrible", "fail", "error", "problem"}

        pos_count = sum(1 for token in doc if token.lemma_.lower() in positive_words)
        neg_count = sum(1 for token in doc if token.lemma_.lower() in negative_words)

        total_words = len([token for token in doc if not token.is_stop and not token.is_punct])

        if total_words == 0:
            return 0.0

        sentiment = (pos_count - neg_count) / total_words
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: Se l'embedding generation fallisce
        """
        try:
            # Use synchronous ollama client embed method
            response = self.ollama_client.embed(
                model="embeddinggemma:300m",
                input=text,
                options={"temperature": 0.0}  # Deterministic embeddings
            )
            return response.embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}") from e

    async def process_memory_entry(self, memory: MemoryEntry) -> MemoryEntry:
        """
        Process complete memory entry con text analysis e embedding.

        Args:
            memory: Memory entry to process

        Returns:
            Processed memory entry con extracted features

        Raises:
            ProcessingError: Se il processing fallisce
        """
        try:
            # Process text
            features = await self.process_text(memory.content, include_embedding=True)

            # Update memory con extracted features
            memory.keywords = features["keywords"]
            memory.entities = features["entities"]
            memory.sentiment = features["sentiment"]
            memory.complexity_score = features["complexity_score"]

            # Set embedding
            if "embedding" in features and features["embedding"]:
                memory.set_embedding(np.array(features["embedding"], dtype=np.float32))

            # Add processing metadata to context
            memory.context_snapshot.update({
                "processing": {
                    "processing_time_ms": features.get("processing_time_ms", 0),
                    "stats": features.get("stats", {}),
                    "pos_tags": features.get("pos_tags", []),
                    "noun_chunks": features.get("noun_chunks", [])
                }
            })

            logger.info(f"Processed memory entry: {memory.id}")
            return memory

        except Exception as e:
            logger.error(f"Memory processing failed for {memory.id}: {e}")
            raise ProcessingError(f"Memory processing failed: {e}") from e

    async def batch_process(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batches per performance.

        Args:
            texts: List of texts to process
            batch_size: Number of texts per batch

        Returns:
            List of processing results

        Raises:
            ProcessingError: Se il batch processing fallisce
        """
        try:
            results = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

                # Process batch concurrently
                batch_tasks = [self.process_text(text) for text in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Handle results e exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process text {i + j}: {result}")
                        results.append({"error": str(result), "index": i + j})
                    else:
                        results.append(result)

            logger.info(f"Batch processing completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise ProcessingError(f"Batch processing failed: {e}") from e

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.nlp:
            # spaCy doesn't require explicit cleanup
            logger.info("Text processor cleanup completed")
        self.nlp = None