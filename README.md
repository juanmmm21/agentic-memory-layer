# Agentic Memory Layer

Persistent, structured memory layer for AI agents that remedies the context limitation of language models (amnesia) via a hybrid architecture: a short-term conversational buffer with automatic synthesis and a long-term episodic store with temporal decay based on the Ebbinghaus forgetting curve.

## Hybrid Memory Architecture

The module implements a coordinated two-tier storage system to balance retrieval speed in active dialogue with the historical retention of relevant facts.

```mermaid
graph TD
    Input[Agent Input / Conversation Turn] --> ShortTerm[Short-Term Memory: Conversational Buffer]
    
    subgraph Short-Term Memory
        ShortTerm --> LengthCheck{Turns > Max Limit?}
        LengthCheck -->|Yes| Summarize[Extract Facts & Compact]
        LengthCheck -->|No| KeepActive[Keep Turns Intact]
        Summarize --> SaveLong[Persist Fact to Long-Term]
        Summarize --> SummaryBuffer[Accumulated Summary History]
    end
    
    subgraph Long-Term Memory (Episodic)
        SaveLong --> VectorGen[Local Embedding Generator]
        VectorGen --> StorageCheck{NanoVectorDB available?}
        StorageCheck -->|Yes| NanoVectorDB[HNSW Index in agent_long_term_memory.json]
        StorageCheck -->|No| MemoryList[Fallback: In-Memory List]
    end
    
    Query[Agent Query / Search] --> RecallFlow[Retrieval Loop - Recall]
    RecallFlow --> SearchVectors[Nearest Vector Search]
    SearchVectors --> Ebbinghaus[Apply Exponential Forgetting Curve]
    Ebbinghaus --> RankResults[Top-K Most Important Memories]
```

### 1. Short-Term Memory (Conversational Buffer)

Keeps the latest interactions in their original message format (`role` and `content`).
*   **Capacity Limit:** The buffer accepts a maximum number of turns ($N_{\text{max}} = 5$ double interactions, i.e. 10 messages total).
*   **Automatic Synthesis Algorithm:** When the limit is exceeded, the system heuristically extracts the key facts from the older turns, leaving the last 2 turns (the most recent) intact in the buffer. The summarized information is appended to the `conversation_summary` string, which is prepended to the agent's prompt, freeing up useful space in the context window without breaking the session's thread of continuity.

### 2. Long-Term Memory (Episodic Decay Memory)

Permanently stores facts, associating each with a feature vector (embedding), a creation timestamp, and a subjective relevance factor.

#### The Exponential Forgetting Curve (Ebbinghaus Heuristic)
When performing a memory search to answer a user query, the system doesn't limit itself to evaluating vector semantic similarity. It applies an exponential temporal decay formula to relevance to mimic biological memory:

$$\text{Recall Score}(F) = \text{Sim}_{\cos}(Q, F) \cdot \text{Imp}(F) \cdot e^{-\lambda \cdot \Delta t}$$

Where:
*   $\text{Sim}_{\cos}(Q, F)$ is the cosine similarity projected onto the range $[0, 1]$ between the query vector $Q$ and the indexed memory vector $F$:
    $$\text{Sim}_{\cos}(Q, F) = \frac{\cos(\mathbf{q}, \mathbf{f}) + 1.0}{2.0}$$
*   $\text{Imp}(F) \in [1, 10]$ is the importance weight originally assigned to the fact when it was stored.
*   $\lambda$ is the forgetting-speed coefficient (`decay_factor`, defaults to `0.05`).
*   $\Delta t = t_{\text{query}} - t_{\text{fact\_creation}}$ represents the memory's age in seconds.

Under this model, old memories with low importance are "forgotten" (their score tends to zero), while critical facts with high importance or extremely recent and semantically related memories dominate the ranking.

#### Deterministic Embedding Generator (Local Word-Hash)
To operate offline with no external dependencies, the module locally computes a unit vector of dimension $D$:
1.  Splits the fact's text into words using the regular expression `\w+`.
2.  Computes the MD5 hash of each word and calculates the corresponding modular index:
    $$h = \text{int(MD5}(w)\text{, 16)} \pmod D$$
3.  Increments the vector's value at position $h$: $V_h \leftarrow V_h + 1.0$.
4.  Applies L2 normalization to the resulting vector to project it onto the surface of a unit hypersphere, allowing the dot product to represent exact cosine similarity.

## Connection to the Ecosystem

This component connects to the infrastructure's modules as follows:
1.  **nano-vector-db:** Acts as the physical store for episodic memory. When `save_fact` is invoked, the memory saves the vector and its metadata using `NanoVectorDB.insert(id, vector, metadata)` and persists the data to disk under the `agent_long_term_memory.json` file.
2.  **orchestra-agents:** Each autonomous agent (Researcher, Writer) uses an instance of this memory to persist its ReAct execution scratchpad and retrieve historical contextual information from past runs.

## Project Structure

*   `memory.py`: Implementation of the `AgenticMemory` class, the short-term buffer with its synthesis algorithm, the local embedding hasher, and the Ebbinghaus temporal scoring engine.
*   `test_memory.py`: Asynchronous unit test suite that validates the buffer's compaction rate, the consistency of the hash embeddings, and the accuracy of the exponential decay rate against elapsed time.
*   `example.py`: Interactive demo simulating the passage of time (30 seconds and 10 minutes) that illustrates how the score of old memories decays and which ones are retained by the agent.

## Long-term store: ids, forgetting, persistence and real encoders

Since `0.1.0` long-term memory is usable as a real store, not only as a demo:

```python
from agentic_memory_layer import AgenticMemory

memory = AgenticMemory(
    embed_fn=my_encoder,        # any text -> vector function (a real embedding model)
    half_life_s=30 * 86400,     # a fact's score halves every 30 days
    path="memory.json",         # loaded if it exists, rewritten atomically on every change
)
with memory.batch():            # bulk inserts write the file once
    fid = memory.save_fact("Prefers answers in Spanish", importance=8, metadata={"source": "chat"})
hits = memory.recall("language preference", top_k=3, where={"source": "chat"}, min_similarity=0.6)
memory.forget(fid)
```

*   `save_fact` returns a stable id (or takes `fact_id`, replacing the fact with that id).
*   `recall` returns `id`, `fact`, `importance`, `recall_score`, `similarity`, `created_at` and
    `metadata`; `where` filters on metadata and `min_similarity` drops weak matches.
*   `vector_store` accepts any object with `insert`/`query` (nano-vector-db's `NanoVectorDB`
    fits as is). If it cannot `delete`, forgotten ids are filtered out at recall time.
*   Without `embed_fn` the old MD5 bag-of-words vector is used, so existing code is unchanged.

## Using it as a package

Since `0.1.0` the code lives in the `agentic_memory_layer` package and the repository is
pip-installable, so other projects can depend on it instead of copying files
or editing `sys.path`:

```bash
pip install "agentic-memory-layer @ git+https://github.com/juanmmm21/agentic-memory-layer.git@v0.1.0"
```

```python
from agentic_memory_layer import AgenticMemory, VectorStore
```

The old top-level modules are kept as thin re-exports, so scripts that add
this folder to `sys.path` keep working. This component imports no other
ai-core component: they are combined in the `aicore` SDK of
[ai-core-infra-hub](https://github.com/juanmmm21/ai-core-infra-hub), which
passes collaborators (vector stores, encoders, tokenizers) in by injection.

Development: `pip install -e ".[dev]"`, then `pytest` and `ruff check .`.

## Installation and Usage

### 1. Initialize the Environment and Install Dependencies

Navigate to the directory and install the required local dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Unit Tests

The suite validates that the temporal decay logic and HNSW persistence exactly match the theory:

```bash
.venv/bin/python -m unittest test_memory.py
```

### 3. Run the Forgetting Curve Demo

Run the demo to visualize how memory scores decay over time:

```bash
.venv/bin/python example.py
```
