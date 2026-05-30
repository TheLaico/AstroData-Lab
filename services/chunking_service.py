"""Estrategias de chunking de texto para el pipeline RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    contenido: str


class ChunkingService:
    """Divide texto en chunks listos para vectorizar."""

    VALID_STRATEGIES = {"fixed", "sentence", "semantic"}

    def dividir(self, texto: str, estrategia: str = "sentence") -> List[TextChunk]:
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("El contenido del documento no puede estar vacio.")

        estrategia_normalizada = (estrategia or "sentence").strip().lower()
        if estrategia_normalizada not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Estrategia de chunking no valida: {estrategia}. "
                f"Valores permitidos: {', '.join(sorted(self.VALID_STRATEGIES))}."
            )

        if estrategia_normalizada == "fixed":
            textos = self._fixed(texto)
        else:
            textos = self._sentence(texto)

        return [
            TextChunk(chunk_id=i, contenido=chunk)
            for i, chunk in enumerate(textos)
            if chunk.strip()
        ]

    def _fixed(self, texto: str, chunk_words: int = 160, overlap_words: int = 24) -> List[str]:
        words = texto.split()
        if len(words) <= chunk_words:
            return [" ".join(words)]

        chunks = []
        step = max(1, chunk_words - overlap_words)
        for start in range(0, len(words), step):
            fragment = words[start:start + chunk_words]
            if fragment:
                chunks.append(" ".join(fragment))
            if start + chunk_words >= len(words):
                break
        return chunks

    def _sentence(self, texto: str, max_words: int = 120, overlap_sentences: int = 1) -> List[str]:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", texto.strip())
            if s.strip()
        ]
        if not sentences:
            return [texto.strip()]

        chunks = []
        current: List[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > max_words:
                chunks.append(" ".join(current).strip())
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_words = sum(len(s.split()) for s in current)
            current.append(sentence)
            current_words += sentence_words

        if current:
            chunks.append(" ".join(current).strip())

        return chunks
