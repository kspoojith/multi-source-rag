# Multilingual RAG System — Complete Flow Explanation

## Input Example: `"Modi ji ka education kya hai?"` (What is Modi's education?)

---

## Step 0 — Server Startup

**File:** `backend/app.py` → `startup()` event

When you run `python -m backend.app`:

1. **`backend/config.py`** loads all settings:
   - `EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"`
   - `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50`
   - `RETRIEVAL_TOP_K = 5`, `SEMANTIC_WEIGHT_ALPHA = 0.7`, `KEYWORD_WEIGHT_BETA = 0.3`
   - `OLLAMA_MODEL = "mistral"`, `OLLAMA_TIMEOUT = 120`
   - Creates directories: `data/raw/`, `data/processed/`, `data/index/`

2. **`startup()` event** in `app.py` triggers:
   - Calls `get_embedding_model()` from `ingestion/embedder.py` → downloads/loads the 471MB multilingual model
   - Calls `indexer.load()` from `ingestion/indexer.py` → reads `data/index/faiss.index` + `metadata.pkl` (21 vectors)

3. FastAPI server starts on `http://0.0.0.0:8000` with HTML UI at `/`

---

## Step 1 — Document Ingestion (POST /ingest)

User clicks **"Ingest Documents"** → `POST /ingest`

### 1a. Load Documents — `ingestion/loader.py`

```
Function: load_documents("data/raw/")
```

- Scans `data/raw/` directory using `os.listdir()`
- Finds 5 files: `modi.txt`, `indian_education.txt`, `constitution.txt`, `isro.txt`, `telangana.txt`
- For each file:
  - Checks extension against `FILE_LOADERS = {".txt": load_text_file, ".json": load_json_file, ".md": load_text_file}`
  - `.txt` match → calls `load_text_file(filepath)` → reads file as UTF-8 string
  - Calls `detect_topic(filename, content)` → scans first 500 chars + filename against `TOPIC_KEYWORDS` dict
    - `"modi.txt"` → keyword `"modi"` found → topic = `"politics"`
    - `"isro.txt"` → keyword `"isro"` found → topic = `"space"`
  - Creates a `Document` dataclass:
    ```python
    Document(
        content="Narendra Damodardas Modi (born 17 September 1950)...",
        metadata={"source": "modi.txt", "topic": "politics"},
        doc_id="modi.txt"
    )
    ```
- **Output:** `List[Document]` with 5 documents

### 1b. Chunk Documents — `ingestion/chunker.py`

```
Function: chunk_documents(documents, chunk_size=500, chunk_overlap=50)
```

For each document, calls `chunk_text()`:

1. **`clean_text(text)`** — normalizes whitespace:
   - `re.sub(r'\n{3,}', '\n\n', text)` → collapses triple+ newlines to double
   - `re.sub(r'[ \t]+', ' ', text)` → collapses multiple spaces to single
   - Strips control characters

2. **Sliding window** splits the cleaned text:
   - `step = chunk_size - chunk_overlap = 500 - 50 = 450`
   - Window 1: characters 0–499 → Chunk 1
   - Window 2: characters 450–949 → Chunk 2 (overlaps 50 chars with Chunk 1)
   - Window 3: characters 900–1399 → Chunk 3
   - ...continues until end of document

3. **Filter** — discards any chunk shorter than `MIN_CHUNK_LENGTH = 30` chars

4. Creates `Chunk` objects:
   ```python
   Chunk(
       text="Narendra Damodardas Modi (born 17 September 1950)...",
       chunk_id="modi.txt_chunk_0",
       metadata={"source": "modi.txt", "topic": "politics", "chunk_index": 0}
   )
   ```

- **Output:** `List[Chunk]` — 21 chunks total from 5 documents

### 1c. Embed Chunks — `ingestion/embedder.py`

```
Function: embedding_model.embed(chunk_texts, normalize=True)
```

- Takes the 21 chunk texts as a list of strings
- Calls `self.model.encode(texts, batch_size=32)`:
  1. **Tokenization** — Multilingual WordPiece tokenizer splits text into subword tokens (handles Hindi/Telugu natively)
  2. **12 Transformer layers** — processes tokens through attention layers
  3. **Mean pooling** — averages all token embeddings into one fixed-size vector
- Each chunk → 384-dimensional float32 vector
- **L2 normalization**: divides each vector by its magnitude → unit vectors (length = 1.0)
  - This is critical because `dot_product(unit_vector_A, unit_vector_B) = cosine_similarity(A, B)`
- **Output:** `np.ndarray` of shape `(21, 384)`

### 1d. Build FAISS Index — `ingestion/indexer.py`

```
Function: indexer.build_index(vectors, metadata)
```

- Creates `faiss.IndexFlatIP(384)` — a brute-force inner-product index
  - "FlatIP" = no compression, exact search, inner product
  - Since vectors are L2-normalized, inner product = cosine similarity
- `self.index.add(vectors)` — adds all 21 vectors in one batch
- Stores `metadata` list in sync: `metadata[0]` ↔ `vector[0]`, etc.

```
Function: indexer.save()
```
- `faiss.write_index()` → saves binary index to `data/index/faiss.index`
- `pickle.dump(metadata)` → saves metadata to `data/index/metadata.pkl`

**Ingestion complete:** 5 docs → 21 chunks → 21 vectors indexed on disk.

---

## Step 2 — Question Answering (POST /ask)

User types **"Modi ji ka education kya hai?"** and clicks **Ask**.

The browser sends `POST /ask {"query": "Modi ji ka education kya hai?"}` to `app.py`.

### 2a. Language Detection — `processing/language_detect.py`

```
Function: detect_language("Modi ji ka education kya hai?")
```

**Step 1: Count Unicode scripts** via `count_script_chars()`:
```
Input: "Modi ji ka education kya hai?"
Result: {devanagari: 0, telugu: 0, tamil: 0, bengali: 0, latin: 23, other: 0}
```
No Devanagari/Telugu → text is in Latin script.

**Step 2: Check for Romanized Hindi** via `detect_romanized_hindi()`:
```
Tokens: ["modi", "ji", "ka", "education", "kya", "hai"]
Check each against ROMANIZED_HINDI_WORDS set:
  "modi"      → NOT in set
  "ji"        → ✅ IN set
  "ka"        → ✅ IN set
  "education" → NOT in set
  "kya"       → ✅ IN set
  "hai"       → ✅ IN set
Hindi count: 4, Total tokens: 6
Ratio: 4/6 = 0.667
```

**Step 3: Decision** — `0.667 > MIXED_THRESHOLD (0.15)` → language = `"mixed"`, confidence = `1.0`

**Output:** `("mixed", 1.0)`

### 2b. Query Normalization — `processing/normalize.py`

```
Function: normalize_query("Modi ji ka education kya hai?")
```

**Step 1:** Calls `detect_language()` → gets `("mixed", 1.0)` (from above)

**Step 2: Transliterate** via `transliterate_query()`:
```
Input tokens: ["Modi", "ji", "ka", "education", "kya", "hai?"]

For each token, look up in TRANSLITERATION_MAP:
  "Modi"      → NOT found → keep as "Modi"
  "ji"        → found → "जी"
  "ka"        → found → "का"
  "education" → NOT found → keep as "education"
  "kya"       → found → "क्या"
  "hai?"      → clean="hai" → found → "है" + trailing "?" → "है?"

Result: "Modi जी का education क्या है?"
```

**Step 3: Expand** — combines original + transliterated:
```
expanded = "Modi ji ka education kya hai? Modi जी का education क्या है?"
```
This is the key insight — the embedding will now capture BOTH the Romanized and Devanagari forms, matching against either script in the corpus.

**Step 4: Remove stopwords** via `remove_stopwords()`:
```
Tokens: ["Modi", "ji", "ka", "education", "kya", "hai?"]
STOPWORDS set contains: "ka", "hai"
After removal: "Modi ji education kya"
(This is used for keyword matching, not for embedding)
```

**Output:**
```python
{
    "original": "Modi ji ka education kya hai?",
    "language": "mixed",
    "confidence": 1.0,
    "normalized": "Modi ji ka education kya hai? Modi जी का education क्या है?",
    "transliterated": "Modi जी का education क्या है?",
    "expanded": "Modi ji ka education kya hai? Modi जी का education क्या है?",
    "stopwords_removed": "Modi ji education kya"
}
```

### 2c. Embed Query — `ingestion/embedder.py`

```
Function: embedding_model.embed_query(expanded_query)
```

- Takes the expanded string: `"Modi ji ka education kya hai? Modi जी का education क्या है?"`
- Tokenizes with multilingual WordPiece → both Latin and Devanagari tokens get proper embeddings
- Passes through 12 transformer layers → mean pooling → L2 normalize
- **Output:** 1D vector of shape `(384,)`, dtype float32

### 2d. FAISS Search — `ingestion/indexer.py`

```
Function: indexer.search(query_vector, top_k=10)
```

- Calls `self.index.search(query_vector, k=10)`:
  - Computes dot product of query vector with ALL 21 stored vectors
  - Since all vectors are L2-normalized, dot product = cosine similarity
  - Returns the 10 highest-scoring vector indices and their scores
- Maps each index `i` to `self.metadata[i]` to get chunk text, source, topic
- **Output:** List of 10 dicts:
  ```python
  [
      {"text": "Modi completed his higher secondary...", "source": "modi.txt",
       "topic": "politics", "score": 0.82, "chunk_id": "modi.txt_chunk_0"},
      {"text": "He enrolled at University of Delhi...", "source": "modi.txt",
       "topic": "politics", "score": 0.76, "chunk_id": "modi.txt_chunk_1"},
      {"text": "NEP 2020 introduced major reforms...", "source": "indian_education.txt",
       "topic": "education", "score": 0.51, "chunk_id": "indian_education.txt_chunk_2"},
      ... # 7 more results with decreasing scores
  ]
  ```

### 2e. Hybrid Search — `retrieval/search.py`

```
Function: hybrid_search(semantic_results, "Modi ji ka education kya hai?", α=0.7, β=0.3)
```

**Step 1: Extract keywords** via `extract_keywords()`:
```
Input: "Modi ji ka education kya hai?"
re.findall(r'[\w]+', text) → ["modi", "ji", "ka", "education", "kya", "hai"]
Filter len >= 3 → ["modi", "education", "kya", "hai"]
```

**Step 2: Score each result** — for each of the 10 semantic results:

```
Result 1: chunk from modi.txt (semantic_score=0.82)
  compute_keyword_score(["modi","education","kya","hai"], chunk_text):
    "modi" in chunk      → ✅ (1)
    "education" in chunk → ✅ (2)
    "kya" in chunk       → ❌ (2)
    "hai" in chunk       → ✅ (3)
    keyword_score = 3/4 = 0.75
  final_score = 0.7 × 0.82 + 0.3 × 0.75 = 0.574 + 0.225 = 0.799
  0.799 >= threshold(0.25) ✅ → KEEP

Result 3: chunk from indian_education.txt (semantic_score=0.51)
  keyword_score = 1/4 = 0.25 (only "education" found)
  final_score = 0.7 × 0.51 + 0.3 × 0.25 = 0.357 + 0.075 = 0.432
  0.432 >= 0.25 ✅ → KEEP
```

**Step 3: Sort** by `final_score` descending → 6 results pass threshold

### 2f. Deduplicate — `retrieval/search.py`

```
Function: deduplicate_results(hybrid_results, similarity_threshold=0.95)
```

- Compares each pair of results by token overlap
- Adjacent chunks from modi.txt (chunk_0 and chunk_1) share 50 chars of overlap
- If token overlap > 95%, the lower-scored one is removed
- **Output:** 2 unique results remain

### 2g. Rerank — `retrieval/rerank.py`

```
Function: rerank_results(results, query_topic=None, max_per_source=3, top_k=5)
```

**Step 1: `rerank_by_topic()`** — if a query topic was detected:
- Checks each result's `topic` field against the detected topic
- Matching topics get `+0.1` boost to `final_score`
- Re-sorts by updated score

**Step 2: `ensure_source_diversity()`**:
- Counts results per source file
- Caps at `max_per_source = 3` from any single file
- With only 2 results, no filtering needed here

**Step 3: Trim** to `top_k = 5` (already under 5)

**Output:** Final 2 chunks ready for the LLM

### 2h. Build Prompt — `generation/prompt.py`

```
Function: build_prompt("Modi ji ka education kya hai?", final_results, "mixed")
```

**Step 1: `build_context_block(results)`** — formats chunks:
```
[Source: modi.txt | Relevance: 0.80]
Modi completed his higher secondary education in Vadnagar.
He then pursued a Master's in Political Science from Gujarat University...

[Source: indian_education.txt | Relevance: 0.43]
India's education system has evolved significantly with NEP 2020...
```

**Step 2:** Combines with `SYSTEM_PROMPT`:
```
System: You are a helpful multilingual question-answering assistant...
STRICT RULES:
1. ONLY use the information provided in the CONTEXT below
2. Do NOT use any external knowledge
3. If context doesn't contain enough info, say so
4. Always mention source file(s)
5. Keep answer concise (2-4 sentences)
...

CONTEXT:
[Source: modi.txt | Relevance: 0.80]
Modi completed his higher secondary education in Vadnagar...

[Source: indian_education.txt | Relevance: 0.43]
India's education system has evolved significantly...

QUESTION: Modi ji ka education kya hai?

Answer (cite sources, be factual):
```

**Step 3:** Adds language instruction — since `lang="mixed"`, adds:
*"The question is in Hindi-English code-mixed. You may respond in the same style."*

**Output:** Complete prompt string

### 2i. LLM Generation — `generation/llm.py`

```
Function: generate_answer(query, results, "mixed")
```

Internally creates an `OllamaLLM` instance and:

**Step 1: `is_available()`** — checks if Ollama is running:
- `GET http://localhost:11434/api/tags` → gets list of installed models
- Finds `"mistral"` in the list → returns `True`

**Step 2: `generate(messages=[...])`** — calls the chat API:
- Sends `POST http://localhost:11434/api/chat`:
  ```json
  {
      "model": "mistral",
      "messages": [
          {"role": "system", "content": "You are a helpful multilingual..."},
          {"role": "user", "content": "CONTEXT:\n[Source: modi.txt]...\n\nQUESTION: Modi ji ka education kya hai?"}
      ],
      "stream": false,
      "options": {"temperature": 0.1}
  }
  ```
- Mistral processes the prompt (~90 seconds on CPU)
- Returns: `"Modi completed his higher secondary education in Vadnagar and later earned an MA in Political Science from Gujarat University. (Source: modi.txt)"`

**Step 3:** Extracts unique source files from results → `["modi.txt", "indian_education.txt"]`

**Output:**
```python
{
    "answer": "Modi completed his higher secondary education in Vadnagar and...",
    "sources": ["modi.txt", "indian_education.txt"],
    "model": "mistral",
    "llm_used": True,
    "latency_ms": 89739
}
```

### 2j. Build Response — `backend/app.py`

The `/ask` endpoint assembles the final `AskResponse`:

```python
AskResponse(
    answer="Modi completed his higher secondary education in Vadnagar...",
    sources=["modi.txt", "indian_education.txt"],
    query_info={
        "original": "Modi ji ka education kya hai?",
        "language": "mixed",
        "language_label": "Code-Mixed (Hindi-English)",
        "confidence": 1.0,
        "normalized": "Modi ji ka education kya hai? Modi जी का education क्या है?",
        "transliterated": "Modi जी का education क्या है?"
    },
    retrieval_info={
        "semantic_results": 10,
        "hybrid_results": 6,
        "deduped_results": 2,
        "final_results": 2,
        "alpha": 0.7, "beta": 0.3
    },
    generation_info={
        "model": "mistral",
        "llm_used": True,
        "normalization_ms": 2,
        "embedding_ms": 362,
        "search_ms": 3,
        "rerank_ms": 1,
        "generation_ms": 89739,
        "total_ms": 90107
    }
)
```

This JSON is sent to the browser, where the JavaScript `renderAnswer(d)` function displays the answer card, query analysis, retrieval pipeline stats, and performance bar.

---

## Complete Data Flow Summary

```
"Modi ji ka education kya hai?"
         │
         ▼
    language_detect.py ──→ "mixed" (0.67 Hindi ratio)
         │
         ▼
    normalize.py ──→ "Modi ji ka education kya hai? Modi जी का education क्या है?"
         │
         ▼
    embedder.py ──→ [0.023, -0.115, 0.089, ... ] (384 floats)
         │
         ▼
    indexer.py ──→ FAISS dot product against 21 vectors → 10 hits
         │
         ▼
    search.py ──→ hybrid scoring (0.7×semantic + 0.3×keyword) → 6 hits
         │
         ▼
    search.py ──→ deduplicate (>95% overlap removed) → 2 hits
         │
         ▼
    rerank.py ──→ topic boost + source diversity → 2 final chunks
         │
         ▼
    prompt.py ──→ anti-hallucination prompt with context blocks
         │
         ▼
    llm.py ──→ Ollama Mistral 7B → grounded answer with source citation
         │
         ▼
    app.py ──→ JSON response with answer + metadata + timings
```

---

## File Responsibility Map

| File | Folder | Role |
|------|--------|------|
| `config.py` | `backend/` | Central settings — all tunable parameters |
| `app.py` | `backend/` | FastAPI server — orchestrates all modules |
| `loader.py` | `ingestion/` | Reads raw files → `Document` objects |
| `chunker.py` | `ingestion/` | Splits documents → overlapping `Chunk` objects |
| `embedder.py` | `ingestion/` | Converts text → 384-dim vectors (MiniLM-L12-v2) |
| `indexer.py` | `ingestion/` | FAISS index — build, save, load, search |
| `language_detect.py` | `processing/` | Detects language: en/hi/te/mixed |
| `normalize.py` | `processing/` | Transliterate + expand + remove stopwords |
| `search.py` | `retrieval/` | Hybrid scoring (semantic + keyword) + dedup |
| `rerank.py` | `retrieval/` | Topic boost + source diversity |
| `prompt.py` | `generation/` | Anti-hallucination prompt construction |
| `llm.py` | `generation/` | Ollama client — sends prompt, gets answer |
| `metrics.py` | `evaluation/` | Recall@k, Precision@k, MRR, hallucination check |
| `benchmarks.py` | `evaluation/` | Ablation study framework |
