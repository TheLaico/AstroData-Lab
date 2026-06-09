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
    """
    Divide texto en chunks listos para vectorizar.

    Estrategias disponibles:
    - fixed    : ventanas de palabras de tamaño fijo con solapamiento.
    - sentence : agrupa oraciones respetando un límite de palabras.
    - semantic : detecta cambios temáticos por similitud léxica entre
                 oraciones contiguas y corta donde la cohesión cae.
    """

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
        elif estrategia_normalizada == "semantic":
            textos = self._semantic(texto)
        else:
            textos = self._sentence(texto)

        return [
            TextChunk(chunk_id=i, contenido=chunk)
            for i, chunk in enumerate(textos)
            if chunk.strip()
        ]

    # ------------------------------------------------------------------
    # Estrategia 1: fixed-size
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Estrategia 2: sentence-based
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Estrategia 3: semantic chunking
    # ------------------------------------------------------------------

    def _semantic(
        self,
        texto: str,
        umbral_similitud: float = 0.25,
        min_oraciones_chunk: int = 2,
        max_words: int = 200,
    ) -> List[str]:
        """
        Detecta cambios temáticos por similitud léxica entre oraciones contiguas.

        Algoritmo:
        1. Divide el texto en oraciones.
        2. Representa cada oración como conjunto de tokens significativos.
        3. Calcula la similitud Jaccard entre oraciones consecutivas.
        4. Corta cuando la similitud cae por debajo de `umbral_similitud`
           y se han acumulado al menos `min_oraciones_chunk` oraciones, o
           cuando el chunk supera `max_words` palabras.

        Args:
            texto: Texto a dividir.
            umbral_similitud: Similitud Jaccard mínima para mantener
                              oraciones en el mismo chunk (0.0–1.0).
            min_oraciones_chunk: Mínimo de oraciones antes de permitir corte.
            max_words: Límite duro de palabras por chunk.

        Returns:
            Lista de strings, cada uno un chunk temáticamente cohesivo.
        """
        oraciones = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", texto.strip())
            if s.strip()
        ]
        if len(oraciones) <= min_oraciones_chunk:
            return [texto.strip()]

        tokens_por_oracion = [self._tokens_semanticos(o) for o in oraciones]

        chunks: List[str] = []
        grupo: List[str] = [oraciones[0]]
        palabras_grupo = len(oraciones[0].split())

        for i in range(1, len(oraciones)):
            similitud = self._jaccard(tokens_por_oracion[i - 1], tokens_por_oracion[i])
            palabras_nueva = len(oraciones[i].split())
            supera_limite = palabras_grupo + palabras_nueva > max_words
            cambio_tematico = similitud < umbral_similitud and len(grupo) >= min_oraciones_chunk

            if supera_limite or cambio_tematico:
                chunks.append(" ".join(grupo).strip())
                grupo = [oraciones[i]]
                palabras_grupo = palabras_nueva
            else:
                grupo.append(oraciones[i])
                palabras_grupo += palabras_nueva

        if grupo:
            chunks.append(" ".join(grupo).strip())

        return chunks

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens_semanticos(texto: str) -> set:
        """Extrae tokens significativos (longitud > 3, sin stopwords)."""
        stopwords = {
            "el", "la", "los", "las", "un", "una", "de", "del", "y", "o",
            "en", "con", "por", "para", "que", "es", "son", "a", "al",
            "se", "su", "sus", "lo", "le", "les", "no", "si", "mas",
        }
        return {
            t
            for t in re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]+", texto.lower())
            if len(t) > 3 and t not in stopwords
        }

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Similitud de Jaccard entre dos conjuntos de tokens."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
