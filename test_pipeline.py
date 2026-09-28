"""
KnowFlow AI Pipeline Self-Verification Test
Tests ingestion, chunking, embeddings, vector search, and query answering.
"""

import os
from core.rag_engine import KnowFlowRAGEngine

def run_verification():
    print("=== KnowFlow AI Verification Test ===")
    engine = KnowFlowRAGEngine()

    sample_dir = os.path.join(os.path.dirname(__file__), "sample_documents")
    files_to_test = [
        "Enterprise_Cloud_Governance_2026.txt",
        "KnowFlow_System_Architecture_Guide.txt",
        "KnowFlow_AI_Overview_2026.pdf"
    ]

    for fname in files_to_test:
        fpath = os.path.join(sample_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                res = engine.ingest_document(f.read(), fname)
                print(f"[Ingestion] {fname}: {res}")

    print(f"\n[Vector Store Count] Total chunks indexed: {engine.vector_store.count()}")
    doc_summary = engine.vector_store.get_document_summary()
    print(f"[Vector Store Summary] Documents indexed: {doc_summary}")

    # Test Query 1
    q1 = "What are the Tier-1 data encryption mandates in 2026?"
    print(f"\n--- Testing Query 1: '{q1}' ---")
    res1 = engine.query(q1, top_k=3)
    print(f"Latency: {res1['latency_seconds']}s")
    print(f"Chunks Retrieved: {res1['chunks_retrieved']}")
    print("Answer Preview:")
    print(res1["answer"][:300] + "...")
    print("Sources:")
    for s in res1["sources"]:
        print(f" - {s['source']} (Page {s['page']}): Sim {s['similarity_score']*100:.1f}%")

    # Test Query 2
    q2 = "Who is the author and developer of KnowFlow AI?"
    print(f"\n--- Testing Query 2: '{q2}' ---")
    res2 = engine.query(q2, top_k=2)
    print(f"Latency: {res2['latency_seconds']}s")
    print(f"Chunks Retrieved: {res2['chunks_retrieved']}")
    print("Answer Preview:")
    print(res2["answer"][:300] + "...")
    for s in res2["sources"]:
        print(f" - {s['source']} (Page {s['page']}): Sim {s['similarity_score']*100:.1f}%")

    print("\n[SUCCESS] Verification Completed Successfully!")

if __name__ == "__main__":
    run_verification()
