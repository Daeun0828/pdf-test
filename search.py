# search.py
import sys, json, argparse, re
import numpy as np
from sentence_transformers import SentenceTransformer

ART = "./artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load():
    E = np.load(f"{ART}/embeddings.npy", allow_pickle=False)  # (N, D)
    chunks = [json.loads(x) for x in open(f"{ART}/chunks.jsonl","r",encoding="utf-8")]
    return E, chunks

def cosine_sim(q, E):
    # q: (D,), E: (N,D) -> returns (N,)
    q = q / (np.linalg.norm(q) + 1e-9)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return En @ q

def pretty_preview(text, max_len=500):
    """잘리는 느낌 줄이기: max_len 이전의 마지막 문장부호까지 자르기."""
    t = text.replace("\n", " ")
    if len(t) <= max_len:
        return t
    cut = t[:max_len]
    # 한국어/영어 문장 끝 후보
    m = re.search(r'.*(다\.|\.|!|\?)', cut)
    return m.group(0) if m else cut + "..."

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="질문(따옴표로 감싸기)")
    ap.add_argument("--topk", type=int, default=10, help="상위 k개 (기본 10)")
    ap.add_argument("--threshold", type=float, default=0.0, help="이 점수 미만은 숨김")
    args = ap.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([args.query], convert_to_numpy=True)[0]

    E, chunks = load()
    if E.size == 0 or E.ndim != 2:
        raise RuntimeError("embeddings.npy가 비었거나 차원이 잘못되었습니다. preprocess_embed.py를 먼저 실행하세요.")

    sims = cosine_sim(q, E)  # (N,)
    order = sims.argsort()[::-1]

    shown = 0
    for idx in order:
        if sims[idx] < args.threshold:
            break
        c = chunks[idx]
        print(f"\n[{shown+1}] score={sims[idx]:.3f}  {c['doc_id']} (chunk {c['chunk_id']})")
        print(pretty_preview(c["text"]))
        shown += 1
        if shown >= args.topk:
            break

    if shown == 0:
        print(f"임계값 {args.threshold} 이상 결과가 없습니다. --threshold를 낮추거나 질의를 바꿔보세요.")
