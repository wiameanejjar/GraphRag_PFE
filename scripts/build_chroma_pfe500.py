import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_corpus(corpus_path: Path, limit: int | None) -> list[dict]:
    with corpus_path.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    if limit and limit > 0:
        return docs[:limit]
    return docs


def to_langchain_docs(raw_docs: list[dict]) -> list[Document]:
    docs = []
    for i, item in enumerate(raw_docs):
        text = item.get("text") or (
            f"Title: {item.get('title', '')}\n\n"
            f"Abstract: {item.get('abstract', '')}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": str(item.get("id", i)),
                    "title": str(item.get("title", "")),
                    "source": "arxiv_cleaned",
                    "row_index": i,
                },
            )
        )
    return docs


def build_index(args: argparse.Namespace) -> dict:
    corpus_path = Path(args.corpus).resolve()
    persist_dir = Path(args.persist_dir).resolve()
    manifest_path = persist_dir / "manifest.json"

    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    raw_docs = load_corpus(corpus_path, args.limit)
    lc_docs = to_langchain_docs(raw_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(lc_docs)

    embeddings = OllamaEmbeddings(
        model=args.embed_model,
        base_url=args.ollama_url,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=args.collection,
    )

    try:
        vectorstore.persist()
    except Exception:
        pass

    stored_count = vectorstore._collection.count()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "persist_dir": str(persist_dir),
        "collection": args.collection,
        "ollama_url": args.ollama_url,
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "limit": args.limit,
        "documents_indexed": len(lc_docs),
        "chunks_created": len(chunks),
        "vectors_stored": stored_count,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible ChromaDB vector baseline for the full PFE corpus."
    )
    parser.add_argument(
        "--corpus",
        default=os.getenv("PFE_CORPUS_PATH", "data/processed/arxiv_cleaned.json"),
        help="Path to arxiv_cleaned.json.",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "indexes/chroma_pfe500_baseline"),
        help="Directory where ChromaDB will be persisted.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "pfe_500_baseline"),
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"),
        help="Ollama server URL. Use http://localhost:11434 outside Docker.",
    )
    parser.add_argument(
        "--embed-model",
        default=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "900")),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "120")),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("NB_DOCS", "0")),
        help="0 means full corpus. Use 500 only if you explicitly want first 500 docs.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing ChromaDB directory before indexing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    manifest = build_index(parse_args())
    print("ChromaDB corpus index built successfully.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
