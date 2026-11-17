import unittest
import time
from agentic_memory_layer.memory import AgenticMemory


class TestAgenticMemory(unittest.TestCase):
    """
    Suite de pruebas unitarias para certificar la memoria de corto plazo (sintesis)
    y la memoria de largo plazo (recuperacion vectorial con decaimiento temporal).
    """

    def setUp(self) -> None:
        import os
        if os.path.exists("agent_long_term_memory.json"):
            try:
                os.remove("agent_long_term_memory.json")
            except OSError:
                pass
        # Configuramos una memoria con decaimiento rapido (factor 0.5) para pruebas controladas
        self.memory = AgenticMemory(short_term_max_turns=2, decay_factor=0.5, vector_dim=16)

    def test_short_term_summarization(self) -> None:
        """
        Verifica que al sobrepasar el maximo de turnos en corto plazo,
        se libere el historial y se genere un resumen acumulado.
        """
        # Añadimos 5 interacciones de usuario y asistente (10 turnos)
        for i in range(5):
            self.memory.add_interaction("user", f"Mensaje del usuario {i} con datos importantes")
            self.memory.add_interaction("assistant", f"Respuesta {i} del asistente")
            
        # El buffer de corto plazo debe haberse recortado a las ultimas 2 interacciones (4 turnos)
        self.assertTrue(len(self.memory.short_term) <= 4)
        # El resumen de la conversacion debe haberse compilado con informacion previa
        self.assertTrue(len(self.memory.conversation_summary) > 0)
        self.assertIn("Mensaje del usuario 0", self.memory.conversation_summary)

    def test_long_term_recall_and_decay(self) -> None:
        """
        Verifica que los hechos de largo plazo se recuperen y decaigan exponencialmente
        con el paso del tiempo simulado.
        """
        t0 = time.time()
        
        # Guardamos un hecho importante en t0
        self.memory.save_fact("Al usuario le encanta programar en Python", importance=8)
        # Guardamos otro hecho de diferente tematica
        self.memory.save_fact("El usuario viaja a Paris en avion", importance=5)
        
        # Consulta inmediata (t = t0)
        results_t0 = self.memory.recall("lenguaje de programacion Python", top_k=1, current_time=t0)
        self.assertEqual(len(results_t0), 1)
        self.assertEqual(results_t0[0]["fact"], "Al usuario le encanta programar en Python")
        score_t0 = results_t0[0]["recall_score"]
        
        # Consulta tardia (t = t0 + 10 segundos de retraso simulado)
        t10 = t0 + 10.0
        results_t10 = self.memory.recall("lenguaje de programacion Python", top_k=1, current_time=t10)
        score_t10 = results_t10[0]["recall_score"]
        
        # El score en t10 debe ser menor debido al decaimiento temporal exponencial de olvido
        self.assertTrue(score_t10 < score_t0)
        
        # Verificamos la tasa de decaimiento teorica: e^(-0.5 * 10) = e^(-5) ≈ 0.0067
        # Por lo tanto score_t10 deberia ser aprox score_t0 * 0.0067
        ratio = score_t10 / score_t0
        self.assertAlmostEqual(ratio, 0.0067379, places=3)


class FakeStore:
    """Base vectorial minima, sin borrado: lo olvidado debe filtrarse al recordar."""

    def __init__(self) -> None:
        self.rows = {}

    def insert(self, id, vector, metadata):
        self.rows[id] = (vector, metadata)

    def query(self, vector, top_k=5):
        scored = sorted(self.rows.items(), key=lambda kv: -sum(a * b for a, b in zip(vector, kv[1][0])))
        return [{"id": k, "metadata": m, "vector": v} for k, (v, m) in scored[:top_k]]


class TestLongTermStore(unittest.TestCase):
    def test_ids_metadata_and_forget(self) -> None:
        memory = AgenticMemory(vector_dim=32)
        fid = memory.save_fact("Juega al padel los martes", importance=6, metadata={"source": "chat"})
        other = memory.save_fact("Vive en Jaen", importance=6)
        hit = memory.recall("padel martes", top_k=1)[0]
        self.assertEqual(hit["id"], fid)
        self.assertEqual(hit["metadata"], {"source": "chat"})
        self.assertTrue(memory.forget(fid))
        self.assertFalse(memory.forget(fid))
        self.assertEqual([h["id"] for h in memory.recall("padel martes", top_k=5)], [other])

    def test_same_id_replaces(self) -> None:
        memory = AgenticMemory(vector_dim=32)
        memory.save_fact("version uno", fact_id="x")
        memory.save_fact("version dos", fact_id="x")
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory.recall("version", top_k=5)[0]["fact"], "version dos")

    def test_injected_embedder_is_normalized(self) -> None:
        memory = AgenticMemory(embed_fn=lambda text: [3.0, 4.0] if "a" in text else [0.0, 2.0])
        self.assertEqual(memory.embed("a"), [0.6, 0.8])
        memory.save_fact("a")
        memory.save_fact("b")
        self.assertEqual(memory.recall("a", top_k=1)[0]["fact"], "a")

    def test_half_life(self) -> None:
        memory = AgenticMemory(half_life_s=86400.0)
        t0 = 1_000_000.0
        memory.save_fact("hecho", importance=4, created_at=t0)
        now = memory.recall("hecho", current_time=t0)[0]["recall_score"]
        later = memory.recall("hecho", current_time=t0 + 86400.0)[0]["recall_score"]
        self.assertAlmostEqual(later / now, 0.5, places=6)

    def test_where_and_min_similarity(self) -> None:
        memory = AgenticMemory(vector_dim=64)
        memory.save_fact("cafe solo por la manana", metadata={"kind": "pref"})
        memory.save_fact("cafe con leche", metadata={"kind": "other"})
        self.assertEqual([h["fact"] for h in memory.recall("cafe", top_k=5, where={"kind": "pref"})], ["cafe solo por la manana"])
        self.assertEqual(memory.recall("zzz qqq", top_k=5, min_similarity=0.9), [])

    def test_persistence_roundtrip_and_batch(self) -> None:
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.json")
            memory = AgenticMemory(vector_dim=32, path=path)
            with memory.batch():
                for i in range(5):
                    memory.save_fact(f"hecho numero {i}", fact_id=f"f{i}")
                self.assertFalse(os.path.exists(path))
            self.assertEqual(len(json.load(open(path))["facts"]), 5)
            memory.forget("f0")
            reloaded = AgenticMemory(vector_dim=32, path=path)
            self.assertEqual(len(reloaded), 4)
            self.assertEqual(reloaded.recall("hecho numero 3", top_k=1)[0]["id"], "f3")

    def test_injected_store_without_delete(self) -> None:
        store = FakeStore()
        memory = AgenticMemory(vector_dim=32, vector_store=store)
        keep = memory.save_fact("perro labrador")
        gone = memory.save_fact("perro pastor")
        memory.forget(gone)
        self.assertIn(gone, store.rows)  # la base no sabe borrar...
        self.assertEqual([h["id"] for h in memory.recall("perro", top_k=5)], [keep])  # ...pero no vuelve


if __name__ == "__main__":
    unittest.main()
