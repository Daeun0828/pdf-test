# preprocess_embed.py
import os, sys, json, argparse, re, unicodedata
from pathlib import Path
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# ── kss 의존성 체크 (문장 단위 청크용)
try:
    import kss
except ImportError:
    print("[오류] kss가 설치되어 있지 않습니다. 먼저 `pip install kss`를 실행하세요.")
    sys.exit(1)

# ── 시끄러운 MuPDF 경고 숨기기
fitz.TOOLS.mupdf_display_errors(False)

# ===== 전처리 유틸 =====
def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return text.replace("\r", "\n")

def fix_hyphen_linebreaks(text: str) -> str:
    # 줄 끝 하이픈으로 끊긴 영어 단어 복원 (regula-\ntion -> regulation)
    return re.sub(r"-\n(?=[A-Za-z0-9])", "", text)

def collapse_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_simple_headers_footers(pages):
    """아주 단순 규칙: 페이지마다 반복되는 첫 줄/마지막 줄 제거"""
    def first_line(s):
        for line in s.splitlines():
            if line.strip(): return line.strip()
        return ""
    def last_line(s):
        for line in reversed(s.splitlines()):
            if line.strip(): return line.strip()
        return ""

    if not pages: return pages
    firsts = [first_line(p) for p in pages]
    lasts  = [last_line(p)  for p in pages]
    common_first = max(set(firsts), key=firsts.count) if firsts else ""
    common_last  = max(set(lasts),  key=lasts.count)  if lasts else ""

    cleaned = []
    for p in pages:
        lines = p.splitlines()
        if lines and lines[0].strip() == common_first:
            lines = lines[1:]
        if lines and lines[-1].strip() == common_last:
            lines = lines[:-1]
        cleaned.append("\n".join(lines))
    return cleaned

def clean_text_per_doc(pages, strip_headers=False):
    pages = [normalize_unicode(x) for x in pages]
    if strip_headers:
        pages = remove_simple_headers_footers(pages)
    text = "\n\n".join(pages)
    text = fix_hyphen_linebreaks(text)
    text = collapse_spaces(text)
    return text

# ===== 문장 분리/청크 =====
def split_sentences_ko(text: str):
    # 한국어 최적화 문장 경계 분리 + 너무 짧은 문장 제거(노이즈 완화)
    sents = [s.strip() for s in kss.split_sentences(text) if s and s.strip()]
    return [s for s in sents if len(s) >= 3]

def chunk_by_sentences(sentences, target_len=800, overlap_sents=2):
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) + 1 <= target_len or not cur:
            cur.append(s); cur_len += len(s) + 1
        else:
            chunks.append(" ".join(cur))
            # 문맥 보존을 위해 마지막 n문장 겹침
            cur = cur[-overlap_sents:] + [s]
            cur_len = sum(len(x) + 1 for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ===== PDF 텍스트 추출 =====
def extract_pages_words(pdf_path: str, max_pages=None, line_merge_tol=2.0):
    """단어 단위로 추출해 줄별 정렬·공백 복원 (붙어쓰기 완화)"""
    pages = []
    with fitz.open(pdf_path) as doc:
        it = list(doc)[:max_pages] if max_pages else doc
        for page in it:
            words = page.get_text("words")  # x0,y0,x1,y1,word, block, line, wno
            words.sort(key=lambda w: (round(w[1], 1), w[0]))  # 줄(y0), x0 기준 정렬
            lines, line, cur_y = [], [], None
            for x0, y0, x1, y1, w, b, ln, wn in words:
                if cur_y is None or abs(y0 - cur_y) < line_merge_tol:
                    line.append(w)
                    cur_y = y0 if cur_y is None else cur_y
                else:
                    if line: lines.append(" ".join(line))
                    line = [w]; cur_y = y0
            if line:
                lines.append(" ".join(line))
            pages.append("\n".join(lines))
    return pages

def extract_pages_text(pdf_path: str, max_pages=None):
    """기본 텍스트 추출 (빠름)"""
    pages = []
    with fitz.open(pdf_path) as doc:
        it = list(doc)[:max_pages] if max_pages else doc
        for page in it:
            pages.append(page.get_text() or "")
    return pages

# ===== 입력 목록 =====
def load_pdf_list(list_file: Path, explicit_list):
    if explicit_list:
        return [str(Path(p).resolve()) for p in explicit_list]
    if not list_file.exists():
        print(f"[오류] {list_file} 가 없습니다. 먼저 scan_pdfs.py로 목록 파일(top5_pdfs.txt)을 만들어주세요.")
        sys.exit(1)
    paths = [line.strip() for line in open(list_file, "r", encoding="utf-8") if line.strip()]
    if not paths:
        print(f"[오류] {list_file} 안에 경로가 없습니다.")
        sys.exit(1)
    return paths

# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser(description="PDF 전처리 + 문장 청크 + 임베딩")
    ap.add_argument("--list-file", default="top5_pdfs.txt", help="PDF 절대경로 목록 파일 (기본: top5_pdfs.txt)")
    ap.add_argument("--pdf", nargs="*", help="목록 파일 대신 직접 PDF 절대경로 지정")
    ap.add_argument("--out", default="artifacts", help="결과 저장 폴더 (기본: artifacts)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer 모델 이름")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap-sents", type=int, default=2, help="문장 겹침 개수")
    ap.add_argument("--min-doc-len", type=int, default=500, help="이 미만 텍스트 길이는 스킵")
    ap.add_argument("--batch-size", type=int, default=64, help="임베딩 배치 크기")
    ap.add_argument("--extract-mode", choices=["words", "text"], default="words",
                    help="PDF 텍스트 추출 방식 (기본 words)")
    ap.add_argument("--max-pages", type=int, default=None, help="문서당 최대 처리 페이지 수 (디버그/속도용)")
    ap.add_argument("--line-merge-tol", type=float, default=2.0, help="words 모드 줄 병합 tolerance")
    ap.add_argument("--strip-headers", action="store_true", help="반복 헤더/푸터 제거")
    args = ap.parse_args()

    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "texts").mkdir(exist_ok=True)

    pdf_list = load_pdf_list(Path(args.list_file), args.pdf)
    print(f"[정보] 처리 대상 {len(pdf_list)}개\n" + "\n".join(f" - {p}" for p in pdf_list))

    model = SentenceTransformer(args.model)

    recs, embs, metas = [], [], []

    for i, pdf in enumerate(pdf_list, 1):
        pdf_path = str(Path(pdf).resolve())
        doc_id = os.path.basename(pdf_path)
        print(f"\n[{i}/{len(pdf_list)}] {doc_id} 처리중...")

        # ── 페이지 추출
        try:
            if args.extract_mode == "words":
                pages = extract_pages_words(pdf_path, max_pages=args.max_pages, line_merge_tol=args.line_merge_tol)
                # words 결과가 비면 text로 폴백
                if not any(p.strip() for p in pages):
                    pages = extract_pages_text(pdf_path, max_pages=args.max_pages)
            else:
                pages = extract_pages_text(pdf_path, max_pages=args.max_pages)
        except Exception as e:
            print(f"  -> 열기 실패: {e} (스킵)")
            continue

        # ── 전처리
        cleaned = clean_text_per_doc(pages, strip_headers=args.strip_headers)
        if len(cleaned) < args.min_doc_len:
            print(f"  -> 텍스트가 너무 적음({len(cleaned)} chars): 스킵")
            continue

        # ── 원문 텍스트 저장(옵션)
        with open(OUT_DIR / "texts" / f"{doc_id}.txt", "w", encoding="utf-8") as f:
            f.write(cleaned)

        # ── 문장 단위 청크
        sentences = split_sentences_ko(cleaned)
        chunks = chunk_by_sentences(sentences, target_len=args.chunk_size, overlap_sents=args.overlap_sents)
        print(f"  -> {len(chunks)} chunks")
        if not chunks:
            print("  -> 청크 0개: 스킵")
            continue

        # ── 임베딩
        vecs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=args.batch_size)

        for j, (c, v) in enumerate(zip(chunks, vecs)):
            recs.append({
                "doc_id": doc_id,
                "chunk_id": j,
                "text": c,
                "source": pdf_path
            })
            embs.append(v)

        metas.append({
            "doc_id": doc_id,
            "path": pdf_path,
            "num_chunks": len(chunks),
            "num_sentences": len(sentences),
            "num_chars": len(cleaned)
        })

    # ── 저장
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if embs:
        np.save(OUT_DIR / "embeddings.npy", np.array(embs, dtype=np.float32))
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print("\n✅ 완료!")
    print(f" - {OUT_DIR / 'chunks.jsonl'}")
    print(f" - {OUT_DIR / 'embeddings.npy'} (벡터 {len(embs)}개)" if embs else " - embeddings.npy (생성 안 됨: 벡터 0개)")
    print(f" - {OUT_DIR / 'meta.json'}")
    print(f" - {OUT_DIR / 'texts'}/*.txt")

if __name__ == "__main__":
    main()
