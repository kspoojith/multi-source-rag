# Multilingual RAG System — Complete Flow Explanation

This document traces the **exact code path** for every query — showing which file,
function, and variable is touched at each stage. Two flows are covered:

1. **Local Knowledge Flow** (`POST /ask`) — answers from ingested corpus files
2. **Open-Domain Web Search Flow** (`POST /ask-web`) — answers from live internet search

---

## Table of Contents

- [Step 0 — Server Startup](#step-0--server-startup)
- [Step 1 — Document Ingestion (POST /ingest)](#step-1--document-ingestion-post-ingest)
- [FLOW A — Local Knowledge (POST /ask)](#flow-a--local-knowledge-post-ask)
  - [Step A1 — Language Detection](#step-a1--language-detection)
  - [Step A2 — Query Normalization](#step-a2--query-normalization)
  - [Step A3 — Embed Query](#step-a3--embed-query)
  - [Step A4 — FAISS Search](#step-a4--faiss-search)
  - [Step A5 — Hybrid Search](#step-a5--hybrid-search)
  - [Step A6 — Deduplicate](#step-a6--deduplicate)
  - [Step A7 — Rerank](#step-a7--rerank)
  - [Step A8 — Build Prompt](#step-a8--build-prompt)
  - [Step A9 — LLM Generation](#step-a9--llm-generation)
  - [Step A10 — Build Response](#step-a10--build-response)
- [FLOW B — Web Search (POST /ask-web)](#flow-b--web-search-post-ask-web)
  - [Step B1 — Language Detection & Normalization](#step-b1--language-detection--normalization)
  - [Step B2 — Query Translation to English](#step-b2--query-translation-to-english)
  - [Step B3 — DuckDuckGo Web Search](#step-b3--duckduckgo-web-search)
  - [Step B4 — Embed Web Chunks On-The-Fly](#step-b4--embed-web-chunks-on-the-fly)
  - [Step B5 — Hybrid Search Within Web Results](#step-b5--hybrid-search-within-web-results)
  - [Step B6 — Web-Aware Prompt & LLM Generation](#step-b6--web-aware-prompt--llm-generation)
  - [Step B7 — Build Web Response](#step-b7--build-web-response)
- [Complete Data Flow Diagrams](#complete-data-flow-diagrams)
- [File Responsibility Map](#file-responsibility-map)

---

## Step 0 — Server Startup

**File:** `backend/app.py` → `startup()` event

When you run `python -m backend.app`:

1. **`backend/config.py`** loads all settings:
   - `EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"`
   - `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50`
   - `RETRIEVAL_TOP_K = 5`, `SEMANTIC_WEIGHT_ALPHA = 0.7`, `KEYWORD_WEIGHT_BETA = 0.3`
   - `OLLAMA_MODEL = "mistral"`, `OLLAMA_TIMEOUT = 120`
   - `WEB_SEARCH_MAX_RESULTS = 8`, `WEB_SEARCH_REGION = "in-en"`
   - Creates directories: `data/raw/`, `data/processed/`, `data/index/`

2. **`startup()` event** in `app.py` triggers:
   - Calls `get_embedding_model()` from `ingestion/embedder.py` → downloads/loads the 471MB multilingual model
   - Calls `indexer.load()` from `ingestion/indexer.py` → reads `data/index/faiss.index` + `metadata.pkl` (21 vectors)

3. FastAPI server starts on `http://0.0.0.0:8000` with HTML UI at `/`
   - The UI has a **Local / Web** mode toggle
   - Local mode sends queries to `POST /ask`
   - Web mode sends queries to `POST /ask-web`

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

# FLOW A — Local Knowledge (POST /ask)

## Input Example: `"Modi ji ka education kya hai?"` (What is Modi's education?)

User types the query and clicks **Ask** (Local Knowledge mode).

The browser sends `POST /ask {"query": "Modi ji ka education kya hai?"}` to `app.py`.

---

### Step A1 — Language Detection

**File:** `processing/language_detect.py`

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

### Step A2 — Query Normalization

**File:** `processing/normalize.py`

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

### Step A3 — Embed Query

**File:** `ingestion/embedder.py`

```
Function: embedding_model.embed_query(expanded_query)
```

- Takes the expanded string: `"Modi ji ka education kya hai? Modi जी का education क्या है?"`
- Tokenizes with multilingual WordPiece → both Latin and Devanagari tokens get proper embeddings
- Passes through 12 transformer layers → mean pooling → L2 normalize
- **Output:** 1D vector of shape `(384,)`, dtype float32

### Step A4 — FAISS Search

**File:** `ingestion/indexer.py`

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

### Step A5 — Hybrid Search

**File:** `retrieval/search.py`

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

### Step A6 — Deduplicate

**File:** `retrieval/search.py`

```
Function: deduplicate_results(hybrid_results, similarity_threshold=0.95)
```

- Compares each pair of results by token overlap
- Adjacent chunks from modi.txt (chunk_0 and chunk_1) share 50 chars of overlap
- If token overlap > 95%, the lower-scored one is removed
- **Output:** 2 unique results remain

### Step A7 — Rerank

**File:** `retrieval/rerank.py`

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

### Step A8 — Build Prompt

**File:** `generation/prompt.py`

```
Function: build_prompt_for_ollama("Modi ji ka education kya hai?", final_results, "mixed")
```

**Step 1: `build_context_block(results)`** — formats chunks:
```
[Source: modi.txt | Relevance: 0.80]
Modi completed his higher secondary education in Vadnagar.
He then pursued a Master's in Political Science from Gujarat University...

[Source: indian_education.txt | Relevance: 0.43]
India's education system has evolved significantly with NEP 2020...
```

**Step 2:** Uses the `SYSTEM_PROMPT`:
```
You are a helpful multilingual question-answering assistant...
STRICT RULES:
1. ONLY use the information provided in the CONTEXT below
2. Do NOT use any external knowledge
3. If context doesn't contain enough info, say so
4. Always mention source file(s)
5. Keep answer concise (2-4 sentences)
...
```

**Step 3:** Adds language instruction — since `lang="mixed"`, adds:
*"The question is in Hindi-English code-mixed. You may respond in the same style."*

**Step 4:** Builds chat messages list:
```python
[
    {"role": "system", "content": "You are a helpful multilingual..."},
    {"role": "user", "content": "CONTEXT:\n[Source: modi.txt]...\n\nQUESTION: Modi ji ka education kya hai?\n\nAnswer concisely using ONLY the context above. Cite source files."}
]
```

### Step A9 — LLM Generation

**File:** `generation/llm.py`

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

### Step A10 — Build Response

**File:** `backend/app.py`

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

# FLOW B — Web Search (POST /ask-web)

## Input Example: `"Elon Musk ki net worth kitni hai?"` (What is Elon Musk's net worth?)

This question **cannot** be answered from local corpus files — it requires real-time web data.

User switches to **🌐 Web Search** mode in the UI and clicks **Search**.

The browser sends `POST /ask-web {"query": "Elon Musk ki net worth kitni hai?"}` to `app.py`.

---

### Step B1 — Language Detection & Normalization

**Files:** `processing/language_detect.py` + `processing/normalize.py`

```
Function: normalize_query("Elon Musk ki net worth kitni hai?")
```

This follows the same Steps A1–A2 from the local flow:

**Language Detection:**
```
Tokens: ["elon", "musk", "ki", "net", "worth", "kitni", "hai"]
Check against ROMANIZED_HINDI_WORDS set:
  "ki"    → ✅ IN set
  "kitni" → ✅ IN set
  "hai"   → ✅ IN set
Hindi count: 3, Total tokens: 7
Ratio: 3/7 = 0.43
0.43 > MIXED_THRESHOLD (0.15) → language = "mixed", confidence = 0.93
```

**Normalization:**
```
Transliterate: "Elon Musk की net worth कितनी है?"
Expand: "Elon Musk ki net worth kitni hai? Elon Musk की net worth कितनी है?"
```

**Output:**
```python
{
    "original": "Elon Musk ki net worth kitni hai?",
    "language": "mixed",
    "confidence": 0.93,
    "normalized": "Elon Musk ki net worth kitni hai? Elon Musk की net worth कितनी है?",
    "transliterated": "Elon Musk की net worth कितनी है?"
}
```

### Step B2 — Query Translation to English

**File:** `processing/translate.py`

```
Function: get_search_query("Elon Musk ki net worth kitni hai?", "mixed")
```

**Why translate?** DuckDuckGo returns MUCH better results for English queries.
"Elon Musk ki net worth kitni hai?" → poor results.
"What is Elon Musk's net worth?" → excellent results from Forbes, Bloomberg, etc.

**Step 1:** Language is not `"en"`, so we proceed with translation.

**Step 2: Try Ollama translation** via `translate_with_ollama()`:
```
1. Check Ollama available: GET http://localhost:11434/api/tags → 200 OK ✅
2. Build translation prompt:
   "Translate the following query to English.
    Only output the English translation, nothing else.
    Keep proper nouns (names, places, organizations) as-is.

    Query: Elon Musk ki net worth kitni hai?

    English translation:"

3. POST http://localhost:11434/api/generate
   model: "mistral"
   stream: false
   temperature: 0.0  (deterministic translation)
   num_predict: 100  (short output)
   timeout: 60s      (capped for translation, not full 120s)

4. Response: "What is Elon Musk's net worth?"
5. Clean up: strip quotes, remove "Translation:" prefix if present
```

**Step 3: Return** — `"What is Elon Musk's net worth?"`

**Fallback path** (if Ollama is unavailable):
```
Function: extract_keywords_for_search("Elon Musk ki net worth kitni hai?")

Tokens: ["Elon", "Musk", "ki", "net", "worth", "kitni", "hai?"]
Remove ALL_STOPWORDS (Hindi + Telugu + English stopwords):
  "ki"    → IN stopwords → REMOVE
  "kitni" → IN stopwords → REMOVE
  "hai?"  → IN stopwords → REMOVE
  
Result: "Elon Musk net worth"
```
Even the fallback produces a decent search query!

### Step B3 — DuckDuckGo Web Search

**File:** `ingestion/web_search.py`

```
Function: search_and_prepare(
    query="Elon Musk ki net worth kitni hai?",
    english_query="What is Elon Musk's net worth?",
    max_results=8
)
```

**Step 1: `search_web()`** — calls DuckDuckGo:
```python
with DDGS() as ddgs:
    results = ddgs.text(
        keywords="What is Elon Musk's net worth?",
        region="in-en",        # India-English for regional relevance
        max_results=8,
        safesearch="moderate"
    )
```

**Step 2:** Parse results → filter out snippets shorter than 20 chars:
```
Result 1: WebResult(
    title="Elon Musk Net Worth | Celebrity Net Worth",
    snippet="Elon Musk is the wealthiest person in the world, with an estimated net worth of...",
    url="https://www.celebritynetworth.com/richest-businessmen/ceos/elon-musk-net-worth/",
    source="celebritynetworth.com"
)
Result 2: WebResult(
    title="Elon Musk - Forbes",
    snippet="Browse today's rankings of the wealthiest people...",
    url="https://www.forbes.com/profile/elon-musk/",
    source="forbes.com"
)
... (6 more results from businessinsider.com, bloomberg.com, etc.)
```

**Step 3: `web_results_to_chunks()`** — convert to RAG-compatible chunk dicts:
```python
[
    {
        "text": "Elon Musk Net Worth | Celebrity Net Worth\nElon Musk is the wealthiest...",
        "source": "celebritynetworth.com",
        "url": "https://www.celebritynetworth.com/...",
        "topic": "web",
        "chunk_id": "web_0"
    },
    {
        "text": "Elon Musk - Forbes\nBrowse today's rankings...",
        "source": "forbes.com",
        "url": "https://www.forbes.com/profile/elon-musk/",
        "topic": "web",
        "chunk_id": "web_1"
    },
    ... # 6 more chunks
]
```

**Step 4: Retry** — if no results (e.g., translation was bad), retries with original Hindi query.

**Output:** `List[Dict]` — 8 web chunks ready for embedding.

**Timing:** ~1,000–2,000ms for DuckDuckGo search.

### Step B4 — Embed Web Chunks On-The-Fly

**File:** `backend/app.py` → `/ask-web` endpoint + `ingestion/embedder.py` + `ingestion/indexer.py`

```python
# Step 1: Embed all web chunk texts
chunk_texts = [c["text"] for c in web_chunks]  # 8 texts
vectors = embedding_model.embed(chunk_texts)    # → shape (8, 384)

# Step 2: Build TEMPORARY FAISS index (NOT saved to disk)
temp_indexer = FAISSIndexer()
temp_indexer.build_index(vectors, web_chunks)
# Creates faiss.IndexFlatIP(384), adds 8 vectors

# Step 3: Embed the normalized query
query_vector = embedding_model.embed_query(normalized_query)
# → shape (384,)

# Step 4: Search within web chunks
semantic_results = temp_indexer.search(query_vector, top_k=8)
# → cosine similarity of query vs each web chunk
```

**Key difference from local flow:** The FAISS index is ephemeral — created per request and discarded. This keeps web results fresh and doesn't pollute the local index.

**Output:** 8 semantic results sorted by cosine similarity to the query.

**Timing:** ~300–600ms for embedding + temporary index creation.

### Step B5 — Hybrid Search Within Web Results

**Files:** `retrieval/search.py` + `retrieval/rerank.py`

The **same hybrid search + rerank pipeline** from the local flow is applied:

```
Step 1: hybrid_search(semantic_results, original_query, α=0.7, β=0.3)
  - Extract keywords: ["elon", "musk", "net", "worth"]
  - For each web chunk:
    final_score = 0.7 × semantic_score + 0.3 × keyword_overlap
  - Filter below 0.25 threshold
  - Sort by final_score → 7 results

Step 2: deduplicate_results(hybrid_results)
  - Compare token overlap between web chunks
  - Remove duplicates (>95% overlap) → 7 results remain
  (Web results are usually unique since they come from different websites)

Step 3: rerank_results(deduped_results, top_k=5)
  - All web chunks have topic="web", so no topic boost applies
  - Source diversity: all from different domains → no filtering needed
  - Trim to top_k=5
```

**Output:** Top 5 web chunks, ranked by relevance.

### Step B6 — Web-Aware Prompt & LLM Generation

**Files:** `generation/prompt.py` + `generation/llm.py`

```
Function: build_web_prompt_for_ollama(query, final_results, "mixed")
```

**Step 1: `build_context_block(results)`** — formats web chunks WITH URLs:
```
[Source: celebritynetworth.com | URL: https://www.celebritynetworth.com/... | Relevance: 0.76]
Elon Musk Net Worth | Celebrity Net Worth
Elon Musk is the wealthiest person in the world, with an estimated net worth of...

[Source: forbes.com | URL: https://www.forbes.com/profile/elon-musk/ | Relevance: 0.73]
Elon Musk - Forbes
Browse today's rankings of the wealthiest people and families globally...

[Source: businessinsider.com | URL: https://www.businessinsider.com/... | Relevance: 0.71]
...
```

**Step 2:** Uses `WEB_SYSTEM_PROMPT` (different from local SYSTEM_PROMPT):
```
You are a helpful multilingual question-answering assistant
that answers questions using web search results.

STRICT RULES:
1. ONLY use the information provided in the WEB SEARCH RESULTS below
2. Do NOT add information beyond what is in the search results
3. If the search results don't have enough info, say so
4. Always mention which website(s) your answer is based on (cite domain names)
5. Keep your answer concise and factual (3-5 sentences maximum)
6. If the question is in Hindi, Telugu, or code-mixed, you may answer in same style
7. Never make up facts not in the search results
```

**Step 3:** Language instruction for "mixed" → "You may respond in Hindi-English mix."

**Step 4: LLM Generation:**
```python
llm = OllamaLLM()
messages = [
    {"role": "system", "content": WEB_SYSTEM_PROMPT},
    {"role": "user", "content": "WEB SEARCH RESULTS:\n[Source: celebritynetworth.com | URL: ...]...\n\nQUESTION: Elon Musk ki net worth kitni hai?\n\nAnswer concisely using ONLY the web search results above. Cite website names."}
]
llm_result = llm.generate(messages=messages)
```

**Response from Mistral:**
```
"Elon Musk ki net worth approximately $638 billion hai,
as per fortune.com."
```

**Fallback** (if Ollama unavailable): Returns raw web snippets with domain names.

### Step B7 — Build Web Response

**File:** `backend/app.py`

The `/ask-web` endpoint assembles the `AskWebResponse`:

```python
AskWebResponse(
    answer="Elon Musk ki net worth approximately $638 billion hai, as per fortune.com.",
    sources=["forbes.com", "celebritynetworth.com", "businessinsider.com", "fortune.com", "aol.com"],
    web_urls=[
        "https://www.celebritynetworth.com/richest-businessmen/ceos/elon-musk-net-worth/",
        "https://www.businessinsider.com/elon-musk-net-worth",
        "https://www.aol.com/articles/elon-musk-net-worth-hits-143107231.html",
        "https://finance.yahoo.com/news/elon-musk-net-worth-large-203110925.html",
        "https://fortune.com/2025/12/16/elon-musk-wealth-soared-past-600-billion..."
    ],
    query_info={
        "original": "Elon Musk ki net worth kitni hai?",
        "language": "mixed",
        "language_label": "Code-Mixed (Hindi-English)",
        "confidence": 0.93,
        "normalized": "Elon Musk ki net worth kitni hai? Elon Musk की net worth कितनी है?",
        "english_query": "What is Elon Musk's net worth?",
        "transliterated": "Elon Musk की net worth कितनी है?"
    },
    search_info={
        "web_results_fetched": 8,
        "semantic_results": 8,
        "hybrid_results": 7,
        "deduped_results": 7,
        "final_results": 5,
        "top_sources": ["celebritynetworth.com", "businessinsider.com", ...],
        "top_scores": [0.758, 0.732, 0.710, 0.688, 0.680]
    },
    generation_info={
        "model": "mistral",
        "llm_used": True,
        "normalization_ms": 2,
        "translation_ms": 15675,
        "web_search_ms": 1292,
        "embedding_ms": 522,
        "rerank_ms": 1,
        "generation_ms": 80720,
        "total_ms": 98212
    }
)
```

The browser's `renderWebAnswer(d)` function displays:
- Answer card with 🌐 Web Search Answer header
- Clickable source URLs as reference links
- Query analysis with the English translation shown
- Web Search Pipeline stats (results fetched → semantic → hybrid → final)
- Performance timing bar (Normalize → Translate → Web Search → Embed+Search → Rerank → Generate)

---

## Complete Data Flow Diagrams

### Flow A — Local Knowledge

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
    prompt.py ──→ SYSTEM_PROMPT + anti-hallucination prompt with context blocks
         │
         ▼
    llm.py ──→ Ollama Mistral 7B → grounded answer with source citation
         │
         ▼
    app.py ──→ JSON response with answer + metadata + timings
```

### Flow B — Web Search

```
"Elon Musk ki net worth kitni hai?"
         │
         ▼
    language_detect.py ──→ "mixed" (0.43 Hindi ratio)
         │
         ▼
    normalize.py ──→ "Elon Musk ki net worth kitni hai? Elon Musk की net worth कितनी है?"
         │
         ▼
    translate.py ──→ "What is Elon Musk's net worth?" (via Ollama or keyword fallback)
         │
         ▼
    web_search.py ──→ DuckDuckGo search → 8 web results (forbes, bloomberg, wiki...)
         │
         ▼
    embedder.py ──→ embed 8 web snippets → shape (8, 384)
         │
         ▼
    indexer.py ──→ TEMPORARY FAISS index → cosine similarity → 8 hits
         │
         ▼
    search.py ──→ hybrid scoring (0.7×semantic + 0.3×keyword) → 7 hits
         │
         ▼
    rerank.py ──→ source diversity → 5 final web chunks
         │
         ▼
    prompt.py ──→ WEB_SYSTEM_PROMPT + web context blocks (with URLs)
         │
         ▼
    llm.py ──→ Ollama Mistral 7B → grounded answer citing website names
         │
         ▼
    app.py ──→ JSON response with answer + web_urls + timings
```

---

## Key Differences: Local vs Web Flow

| Aspect | Local (`/ask`) | Web (`/ask-web`) |
|--------|---------------|-----------------|
| **Data Source** | Pre-ingested corpus files | Live DuckDuckGo search results |
| **Requires /ingest** | Yes — index must be built first | No — works without any local data |
| **Translation Step** | Not needed | Translates query to English for better search |
| **FAISS Index** | Persistent (saved to disk) | Temporary (per-request, discarded) |
| **Context Block** | `[Source: filename.txt]` | `[Source: domain.com | URL: https://...]` |
| **System Prompt** | `SYSTEM_PROMPT` (cite files) | `WEB_SYSTEM_PROMPT` (cite websites) |
| **Source Citations** | File names: `modi.txt` | Domain names: `forbes.com` |
| **Response Links** | None | Clickable URLs to original web pages |
| **Latency** | ~90s (mostly LLM) | ~100s (translation + search + LLM) |
| **New Files Used** | — | `translate.py`, `web_search.py` |

---

## File Responsibility Map

| File | Folder | Role | Used By |
|------|--------|------|---------|
| `config.py` | `backend/` | Central settings — all tunable parameters | Both flows |
| `app.py` | `backend/` | FastAPI server — `/`, `/ask`, `/ask-web`, `/ingest`, `/health` | Both flows |
| `loader.py` | `ingestion/` | Reads raw files → `Document` objects | Local only |
| `chunker.py` | `ingestion/` | Splits documents → overlapping `Chunk` objects | Local only |
| `embedder.py` | `ingestion/` | Converts text → 384-dim vectors (MiniLM-L12-v2) | Both flows |
| `indexer.py` | `ingestion/` | FAISS index — build, save, load, search | Both flows |
| `web_search.py` | `ingestion/` | DuckDuckGo search → web chunks | Web only |
| `language_detect.py` | `processing/` | Detects language: en/hi/te/mixed | Both flows |
| `normalize.py` | `processing/` | Transliterate + expand + remove stopwords | Both flows |
| `translate.py` | `processing/` | Translates Hindi/Telugu queries to English for search | Web only |
| `search.py` | `retrieval/` | Hybrid scoring (semantic + keyword) + dedup | Both flows |
| `rerank.py` | `retrieval/` | Topic boost + source diversity | Both flows |
| `prompt.py` | `generation/` | SYSTEM_PROMPT + WEB_SYSTEM_PROMPT + context blocks | Both flows |
| `llm.py` | `generation/` | Ollama client — sends prompt, gets answer | Both flows |
| `metrics.py` | `evaluation/` | Recall@k, Precision@k, MRR, hallucination check | Evaluation |
| `benchmarks.py` | `evaluation/` | Ablation study framework | Evaluation |
