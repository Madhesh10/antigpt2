# chat/ingest.py
"""
RAG ingestion + answering module:
- embeddings: OpenAI
- generation: DeepSeek (preferred) with OpenAI fallback
"""

import os
import json
import traceback
from pathlib import Path
import pickle
import requests

try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss
except Exception:
    faiss = None

from django.conf import settings
from .models import Document

# -------------------------
# Config (env)
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

DEEPSEEK_GEN_URL = os.environ.get(
    "DEEPSEEK_GEN_URL", "https://api.deepseek.com/v1/chat/completions"
)

# model defaults — override with env vars if needed
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DEEPSEEK_GEN_MODEL = os.environ.get("DEEPSEEK_GEN_MODEL", "deepseek-chat")

STORE_DIR = Path(settings.BASE_DIR) / "vectorstores"
STORE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# optional openai client detection
# -------------------------
openai_pkg = None
OpenAIClient = None
try:
    import openai as openai_pkg  # classic openai package
except Exception:
    openai_pkg = None

try:
    # new-style client (openai >= 1.0)
    from openai import OpenAI as OpenAIClient  # type: ignore
except Exception:
    OpenAIClient = None

# -------------------------
# Helper headers
# -------------------------
def _deepseek_headers():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")
    return {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

# -------------------------
# OpenAI embeddings helper
# -------------------------
def get_embeddings_from_openai(texts):
    """
    texts: list[str] or str
    returns: list[list[float]]
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    # prefer modern OpenAIClient if available
    if OpenAIClient is not None:
        try:
            client = OpenAIClient(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
            # resp.data is list-like
            embeddings = [item.embedding for item in resp.data]
            if not embeddings:
                raise RuntimeError("No embeddings returned from OpenAI (client).")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings (OpenAIClient) failed: {e}") from e

    # fallback to classic openai package style
    if openai_pkg is not None:
        try:
            openai_pkg.api_key = OPENAI_API_KEY
            resp = openai_pkg.Embedding.create(model=OPENAI_EMBED_MODEL, input=texts)
            # resp["data"] is list with "embedding"
            embeddings = [item["embedding"] for item in resp["data"]]
            if not embeddings:
                raise RuntimeError("No embeddings returned from OpenAI (legacy).")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings (legacy) failed: {e}") from e

    # if we reach here, user lacks openai SDK
    raise RuntimeError("openai package not installed. Install it or configure OpenAIClient.")


# -------------------------
# DeepSeek generation helper
# -------------------------
def generate_with_deepseek(prompt, max_tokens=300):
    """
    Calls DeepSeek chat/completions API (OpenAI-compatible shape).
    Raises RuntimeError on problems (so callers can fallback to OpenAI).
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set for DeepSeek generation")

    payload = {
        "model": DEEPSEEK_GEN_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(DEEPSEEK_GEN_URL, json=payload, headers=_deepseek_headers(), timeout=60)
    except Exception as e:
        raise RuntimeError(f"DeepSeek generation network error: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek generation failed {resp.status_code}: {resp.text}")

    data = resp.json()

    # OpenAI-like extraction
    if isinstance(data, dict):
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            c0 = data["choices"][0]
            msg = c0.get("message")
            if isinstance(msg, dict):
                txt = msg.get("content") or msg.get("text")
                if txt:
                    return txt
            txt = c0.get("text") or c0.get("output")
            if txt:
                return txt if isinstance(txt, str) else json.dumps(txt)
        text = data.get("output") or data.get("answer") or data.get("text")
        if isinstance(text, list):
            return "\n".join(text)
        if text:
            return text
    return str(data)


# -------------------------
# OpenAI generation helper (fallback)
# -------------------------
def generate_with_openai(prompt, max_tokens=300, model="gpt-4o-mini"):
    """
    Generate with OpenAI. Works with new and legacy openai Python SDKs.
    Raises RuntimeError on failure.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured for OpenAI fallback.")

    # modern OpenAIClient
    if OpenAIClient is not None:
        try:
            client = OpenAIClient(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(model=model, messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ], max_tokens=max_tokens, temperature=0.2)
            # try to extract text
            if hasattr(resp, "choices") and resp.choices:
                ch = resp.choices[0]
                # new client may use message.content
                if isinstance(ch, dict):
                    msg = ch.get("message")
                    if isinstance(msg, dict):
                        return msg.get("content") or msg.get("text") or str(ch)
                # fallback
                return str(ch)
            return str(resp)
        except Exception as e:
            raise RuntimeError(f"OpenAI (OpenAIClient) failed: {e}") from e

    # legacy openai package
    if openai_pkg is not None:
        try:
            openai_pkg.api_key = OPENAI_API_KEY
            resp = openai_pkg.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            if "choices" in resp and resp["choices"]:
                ch = resp["choices"][0]
                if "message" in ch and isinstance(ch["message"], dict):
                    return ch["message"].get("content") or ch["message"].get("text") or str(ch)
                return ch.get("text") or str(ch)
            return str(resp)
        except Exception as e:
            raise RuntimeError(f"OpenAI (legacy) failed: {e}") from e

    raise RuntimeError("OpenAI SDK not installed; cannot call OpenAI fallback.")


# -------------------------
# Text extraction + chunking (unchanged)
# -------------------------
def extract_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    text = ""
    if ext == ".pdf":
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    text += (p.extract_text() or "") + "\n"
        except Exception as e:
            print("PDF extract error:", e)
    elif ext in (".docx", ".doc"):
        try:
            from docx import Document as DocxDoc
            d = DocxDoc(path)
            for p in d.paragraphs:
                text += p.text + "\n"
        except Exception as e:
            print("DOCX extract error:", e)
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print("Text extract error:", e)
    return text or ""


def chunk_text(text: str, chunk_size=800, overlap=100):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# -------------------------
# Ingest document (embeddings -> store)
# -------------------------
def user_index_paths(user_id: int):
    return STORE_DIR / f"user_{user_id}.index", STORE_DIR / f"user_{user_id}_meta.pkl"


def ingest_document_file(doc: Document):
    """
    doc: chat.models.Document instance
    Extract text -> chunks -> embeddings -> save to FAISS (or fallback).
    """
    try:
        file_path = Path(doc.file.path)
    except Exception as e:
        print("ingest_document_file: cannot access file path:", e)
        doc.processed = False
        doc.save()
        return

    text = extract_text_from_file(file_path)
    if not text:
        doc.processed = True
        doc.save()
        return

    chunks = chunk_text(text)
    if not chunks:
        doc.processed = True
        doc.save()
        return

    # ===== use OpenAI for embeddings =====
    embeddings = get_embeddings_from_openai(chunks)
    if not embeddings:
        doc.processed = True
        doc.save()
        return

    emb_dim = len(embeddings[0])
    idx_path, meta_path = user_index_paths(doc.user.id)

    if faiss and np is not None:
        # write/read faiss index
        if idx_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
            except Exception:
                index = faiss.IndexFlatL2(emb_dim)
                metas = []
        else:
            index = faiss.IndexFlatL2(emb_dim)
            metas = []

        arr = np.array(embeddings).astype("float32")
        index.add(arr)

        if meta_path.exists():
            metas = pickle.load(open(meta_path, "rb"))
        else:
            metas = []

        for ch in chunks:
            metas.append({"text": ch, "doc_id": doc.id, "title": doc.title})

        faiss.write_index(index, str(idx_path))
        pickle.dump(metas, open(meta_path, "wb"))
    else:
        # fallback: save vectors in meta
        if meta_path.exists():
            metas = pickle.load(open(meta_path, "rb"))
        else:
            metas = []
        for emb, ch in zip(embeddings, chunks):
            metas.append({"vector": emb, "text": ch, "doc_id": doc.id, "title": doc.title})
        pickle.dump(metas, open(meta_path, "wb"))

    doc.processed = True
    doc.save()


# -------------------------
# Retrieval + answering
# -------------------------
def retrieve_top_k(user, query, k=4):
    q_emb = get_embeddings_from_openai([query])[0]
    idx_path, meta_path = user_index_paths(user.id)
    if not meta_path.exists():
        return []

    # faiss if exists
    if faiss and idx_path.exists() and np is not None:
        try:
            index = faiss.read_index(str(idx_path))
            q_arr = (np.array([q_emb])).astype("float32")
            D, I = index.search(q_arr, k)
            with open(meta_path, "rb") as f:
                metas = pickle.load(f)
            results = []
            for idx in I[0]:
                if idx < len(metas):
                    results.append(metas[idx]["text"])
            return results
        except Exception as e:
            print("retrieve_top_k (faiss) error:", e)

    # fallback using saved metas and numpy cosine similarity
    with open(meta_path, "rb") as f:
        metas = pickle.load(f)

    if np is None:
        return [m["text"] for m in metas[:k]]

    vectors = []
    texts = []
    for m in metas:
        v = m.get("vector") or m.get("embedding")
        if v:
            vectors.append(np.array(v).astype("float32"))
            texts.append(m["text"])
    if not vectors:
        return []
    mat = np.vstack(vectors)
    qv = np.array(q_emb).astype("float32")
    dot = mat @ qv
    denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(qv) + 1e-12))
    sims = dot / denom
    top_idx = np.argsort(-sims)[:k]
    return [texts[i] for i in top_idx if i < len(texts)]


# ---------- helper: try LLM generation (DeepSeek preferred, OpenAI fallback) ----------
def call_llm_fallback(prompt, max_tokens=300):
    """
    Try to call DeepSeek generation endpoint first; if it fails and OPENAI_API_KEY exists,
    call OpenAI instead.
    Returns string answer or raises RuntimeError.
    """
    # try DeepSeek generation if configured
    try:
        if DEEPSEEK_API_KEY:
            return generate_with_deepseek(prompt, max_tokens=max_tokens)
    except Exception as e:
        print("generate_with_deepseek failed (will try OpenAI):", e)
        traceback.print_exc()

    # fallback to OpenAI if configured
    if OPENAI_API_KEY:
        try:
            return generate_with_openai(prompt, max_tokens=max_tokens)
        except Exception as e:
            print("generate_with_openai failed:", e)
            traceback.print_exc()
            raise RuntimeError("All LLM generation attempts failed: " + str(e))

    # if we reach here, no generation available
    raise RuntimeError("No generation backend available (DeepSeek failed and OPENAI_API_KEY not set)")


# ---------- replacement answer_question ----------
def answer_question(user, question):
    """
    Combined RAG + general LLM fallback.

    Flow:
      1. Retrieve top-k contexts for the user (RAG)
      2. If contexts found, build a prompt that includes the context and the question;
         else, send the plain question to the LLM fallback.
      3. Return the LLM answer string.
    """
    # 1) retrieve contexts (may be empty)
    try:
        contexts = retrieve_top_k(user, question, k=4)
    except Exception as e:
        print("retrieve_top_k error:", e)
        traceback.print_exc()
        contexts = []

    # 2) Build prompt
    if contexts:
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the context does not contain the answer, answer from your general knowledge.\n\n"
            "Context:\n"
        )
        for i, c in enumerate(contexts):
            prompt += f"\n--- Context {i+1} ---\n{c}\n"
        prompt += f"\nQuestion: {question}\nAnswer concisely:"
    else:
        prompt = f"You are a helpful assistant. Answer the question:\n\nQuestion: {question}\nAnswer concisely:"

    # 3) Call generation (DeepSeek preferred, OpenAI fallback)
    try:
        answer = call_llm_fallback(prompt, max_tokens=300)
    except Exception as e:
        print("LLM generation error:", e)
        traceback.print_exc()
        return f"Error generating answer: {e}"

    # 4) Return answer
    return answer or "No answer returned by generation backend."
