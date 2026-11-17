"""Capa de memoria para agentes: corto plazo resumido y largo plazo episódico con olvido.

Largo plazo: cada hecho guarda su texto, su importancia, su vector y sus metadatos; ``recall``
puntúa ``similitud × importancia × e^(-λ·Δt)`` (Ebbinghaus). Los hechos tienen id estable, se
pueden olvidar y se persisten en un JSON si se da ``path``. El codificador y la base vectorial
se inyectan: este paquete no importa ningún otro componente del ecosistema.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Sequence[float]]
STORE_VERSION = 1


class VectorStore(Protocol):
    """Base vectorial opcional para la memoria de largo plazo.

    ``nano_vector_db.NanoVectorDB`` la cumple tal cual, pero este paquete no la importa: los
    componentes del ecosistema no se importan entre si y quien los une (el SDK ``aicore`` del
    hub) pasa la instancia al construir la memoria. Si la base no sabe borrar, lo olvidado se
    filtra al recuperar.
    """

    def insert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None: ...

    def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]: ...


class AgenticMemory:
    """
    Capa de memoria persistente y estructurada para agentes de Inteligencia Artificial.

    Combina:
    1. Memoria de Corto Plazo (Conversational Buffer): Almacena las ultimas interacciones
       aplicando un algoritmo de sintesis/resumen automatico al sobrepasar limites.
    2. Memoria de Largo Plazo (Episodic Decay Memory): Guarda hechos persistentes con
       puntuaciones de importancia y formula de decaimiento exponencial-temporal de olvido.

    Args:
        short_term_max_turns: pares usuario/asistente que caben antes de resumir.
        decay_factor: λ por segundo. Ignorado si se da ``half_life_s``.
        vector_dim: dimension del vector de hash (solo sin ``embed_fn``).
        vector_store: base vectorial externa; sin ella, busqueda exacta en memoria.
        embed_fn: codificador texto -> vector (p. ej. un modelo real); sin el, hash de palabras.
        half_life_s: segundos en que la puntuacion de un hecho se reduce a la mitad.
        path: fichero JSON donde se persisten los hechos (se carga si existe).
    """

    def __init__(
        self,
        short_term_max_turns: int = 5,
        decay_factor: float = 0.05,
        vector_dim: int = 64,
        vector_store: Optional[VectorStore] = None,
        embed_fn: Optional[EmbedFn] = None,
        half_life_s: Optional[float] = None,
        path: Optional[str] = None,
    ) -> None:
        self.max_turns = short_term_max_turns
        if half_life_s is not None:
            if half_life_s <= 0:
                raise ValueError("half_life_s debe ser positivo")
            decay_factor = math.log(2) / half_life_s
        self.decay_factor = decay_factor
        self.vector_dim = vector_dim
        self.embed_fn = embed_fn
        self.path = path

        # Buffer de corto plazo
        self.short_term: List[Dict[str, str]] = []
        self.conversation_summary: str = ""

        # Almacenamiento de largo plazo: la base vectorial inyectada o, sin ella, una lista.
        self.vector_db = vector_store
        self.use_real_db = vector_store is not None

        # Lista local: la copia de verdad de todo lo guardado (es lo que se persiste).
        self.fallback_db: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._forgotten: set = set()
        self._lock = threading.RLock()
        self._deferred = 0

        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------ corto plazo

    def add_interaction(self, role: str, content: str) -> None:
        """Añade un turno al buffer de corto plazo y gatilla resumen si excede max_turns."""
        self.short_term.append({"role": role, "content": content})

        # Si excedemos el limite de turnos (par de usuario-asistente cuenta como 2 turnos)
        if len(self.short_term) > self.max_turns * 2:
            self._summarize_short_term()

    def _summarize_short_term(self) -> None:
        """
        Sintetiza la conversacion acumulada en un resumen para liberar espacio de contexto.
        En produccion, esto llamaria a un LLM. Aqui implementamos un extractor secuencial de eventos.
        """
        turns_to_summarize = self.short_term[:-2]  # Dejamos las ultimas 2 interacciones intactas
        self.short_term = self.short_term[-2:]

        extracted_facts = []
        for turn in turns_to_summarize:
            content = turn["content"]
            if turn["role"] == "user" and len(content) > 10:
                extracted_facts.append(f"El usuario pregunto o menciono: '{content}'")
            elif turn["role"] == "assistant" and len(content) > 10:
                extracted_facts.append(f"El agente respondio: '{content[:50]}...'")

        prefix = f"{self.conversation_summary}\n" if self.conversation_summary else ""
        self.conversation_summary = prefix + " -> " + " | ".join(extracted_facts)
        logger.info(f"Memoria a corto plazo resumida. Historial acumulado: {len(self.conversation_summary)} caracteres.")

    # ------------------------------------------------------------------ vectores

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Genera un vector unitario deterministicamente basado en una bolsa de palabras
        hasheadas con MD5, garantizando alta similitud solo ante coincidencia de terminos.
        """
        words = re.findall(r"\w+", text.lower())
        vector = [0.0] * self.vector_dim
        if not words:
            return vector

        for w in words:
            h_val = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            h = h_val % self.vector_dim
            vector[h] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, text: str) -> List[float]:
        """El vector de ``text`` con ``embed_fn`` (normalizado) o, sin el, con el hash."""
        if self.embed_fn is None:
            return self._generate_mock_embedding(text)
        vector = [float(v) for v in self.embed_fn(text)]
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm > 0 else vector

    # ------------------------------------------------------------------ largo plazo

    def save_fact(
        self,
        fact: str,
        importance: float = 5,
        metadata: Optional[Dict[str, Any]] = None,
        fact_id: Optional[str] = None,
        created_at: Optional[float] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> str:
        """
        Persiste un hecho relevante en la memoria episódica de largo plazo.

        Args:
            fact: Hecho a memorizar.
            importance: Calificacion de relevancia emocional o pragmatica (1 a 10).
            metadata: datos del llamador (origen, ruta...), devueltos al recordar.
            fact_id: id estable; si ya existe, el hecho se sustituye.
            created_at: marca de tiempo (por defecto, ahora).
            embedding: vector ya calculado (se ahorra llamar a ``embed_fn``).

        Returns:
            El id del hecho.
        """
        timestamp = time.time() if created_at is None else float(created_at)
        vector = list(embedding) if embedding is not None else self.embed(fact)
        fid = fact_id or str(uuid.uuid4())

        fact_item = {
            "id": fid,
            "fact": fact,
            "importance": importance,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "embedding": vector,
            "metadata": dict(metadata or {}),
        }

        with self._lock:
            if fid in self._by_id:
                self._drop(fid)
            self._forgotten.discard(fid)
            self.fallback_db.append(fact_item)
            self._by_id[fid] = fact_item
            if self.use_real_db and self.vector_db is not None:
                try:
                    self.vector_db.insert(
                        id=fid,
                        vector=vector,
                        metadata={k: v for k, v in fact_item.items() if k != "embedding"},
                    )
                except Exception as e:
                    logger.error(f"Fallo al insertar en la base vectorial: {str(e)}. Queda en la lista local.")
            self._autosave()

        logger.info(f"Hecho guardado en memoria de largo plazo (Importancia: {importance}): '{fact[:80]}'")
        return fid

    def _drop(self, fid: str) -> None:
        self.fallback_db = [item for item in self.fallback_db if item["id"] != fid]
        self._by_id.pop(fid, None)

    def forget(self, fact_id: str) -> bool:
        """Olvida un hecho por su id. Devuelve si existia."""
        with self._lock:
            existed = fact_id in self._by_id
            self._drop(fact_id)
            if self.use_real_db and self.vector_db is not None:
                delete = getattr(self.vector_db, "delete", None)
                if callable(delete):
                    try:
                        delete(fact_id)
                    except Exception as e:  # la base puede no tenerlo: se filtra al recordar
                        logger.warning(f"La base vectorial no pudo borrar {fact_id}: {e}")
                self._forgotten.add(fact_id)
            if existed:
                self._autosave()
            return existed

    def facts(self) -> List[Dict[str, Any]]:
        """Todos los hechos guardados (sin los vectores)."""
        with self._lock:
            return [{k: v for k, v in item.items() if k != "embedding"} for item in self.fallback_db]

    def __len__(self) -> int:
        return len(self._by_id)

    def score(self, similarity: float, importance: float, created_at: float, current_time: float) -> float:
        """``similitud [0,1] × importancia × e^(-λ·Δt)``."""
        delta_t = max(0.0, current_time - created_at)
        return similarity * importance * math.exp(-self.decay_factor * delta_t)

    def recall(
        self,
        query: str,
        top_k: int = 3,
        current_time: Optional[float] = None,
        min_similarity: float = 0.0,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recupera los top_k recuerdos mas relevantes aplicando la formula de decaimiento temporal (olvido).

        Formula de puntuacion de recuerdo:
            Score = Similitud_Coseno * Importancia * e^(-lambda * delta_t)

        Args:
            min_similarity: descarta hechos cuya similitud (en [0, 1]) no la alcance.
            where: solo hechos cuyos metadatos coinciden con todas estas claves.
        """
        if current_time is None:
            current_time = time.time()

        query_vector = self.embed(query)

        with self._lock:
            items: List[Dict[str, Any]] = []
            if self.use_real_db and self.vector_db is not None:
                try:
                    results = self.vector_db.query(vector=query_vector, top_k=max(20, top_k * 4))
                    for res in results:
                        meta = res["metadata"]
                        fid = str(meta.get("id", res.get("id")))
                        if fid in self._forgotten or fid not in self._by_id:
                            continue
                        items.append(self._by_id[fid])
                except Exception as e:
                    logger.error(f"Fallo en consulta a la base vectorial: {str(e)}. Usando la lista local.")
                    items = list(self.fallback_db)
            else:
                items = list(self.fallback_db)

            candidates: List[Tuple[float, float, Dict[str, Any]]] = []
            for item in items:
                if where and any(item["metadata"].get(k) != v for k, v in where.items()):
                    continue
                cosine = sum(qv * iv for qv, iv in zip(query_vector, item["embedding"]))
                # Ajustamos rango de similitud de [-1, 1] a [0, 1]
                similarity = (cosine + 1.0) / 2.0
                if similarity < min_similarity:
                    continue
                recall_score = self.score(similarity, item["importance"], item["created_at"], current_time)
                candidates.append((recall_score, similarity, item))

            candidates.sort(key=lambda x: x[0], reverse=True)

            recalled_items = []
            for score, similarity, item in candidates[:top_k]:
                item["last_accessed"] = current_time
                recalled_items.append({
                    "id": item["id"],
                    "fact": item["fact"],
                    "importance": item["importance"],
                    "recall_score": float(score),
                    "similarity": float(similarity),
                    "created_at": item["created_at"],
                    "metadata": dict(item["metadata"]),
                })
            return recalled_items

    # ------------------------------------------------------------------ persistencia

    def save(self, path: Optional[str] = None) -> None:
        """Escribe los hechos en JSON de forma atómica (fichero temporal + rename)."""
        target = path or self.path
        if not target:
            raise ValueError("No hay ruta donde guardar la memoria")
        with self._lock:
            payload = {
                "version": STORE_VERSION,
                "decay_factor": self.decay_factor,
                "summary": self.conversation_summary,
                "facts": self.fallback_db,
            }
            directory = os.path.dirname(os.path.abspath(target))
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".memory-", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, target)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

    def load(self, path: Optional[str] = None) -> None:
        """Carga los hechos de un JSON escrito por ``save`` (sustituye a los actuales)."""
        source = path or self.path
        if not source:
            raise ValueError("No hay ruta de la que cargar la memoria")
        with open(source, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != STORE_VERSION:
            raise ValueError(f"Version de memoria desconocida: {payload.get('version')}")
        with self._lock:
            self.fallback_db = []
            self._by_id = {}
            self.conversation_summary = payload.get("summary", "")
            for item in payload.get("facts", []):
                item.setdefault("metadata", {})
                self.fallback_db.append(item)
                self._by_id[item["id"]] = item
                if self.use_real_db and self.vector_db is not None:
                    self.vector_db.insert(
                        id=item["id"],
                        vector=item["embedding"],
                        metadata={k: v for k, v in item.items() if k != "embedding"},
                    )

    @contextlib.contextmanager
    def batch(self) -> Iterator["AgenticMemory"]:
        """Agrupa muchas escrituras: el JSON se reescribe una sola vez, al salir."""
        with self._lock:
            self._deferred += 1
        try:
            yield self
        finally:
            with self._lock:
                self._deferred -= 1
                if self._deferred == 0:
                    self._autosave()

    def _autosave(self) -> None:
        if self.path and not self._deferred:
            self.save(self.path)
