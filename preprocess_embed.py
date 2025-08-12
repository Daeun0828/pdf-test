# preprocess_embed.py - 최적화 버전
import os, sys, json, argparse, re, unicodedata
from pathlib import Path
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import gc
from time import time
from typing import List, Dict, Tuple, Optional

# ── kss 의존성 체크 (문장 단위 청크용)
try:
    import kss
except ImportError:
    print("[오류] kss가 설치되어 있지 않습니다. 먼저 `pip install kss`를 실행하세요.")
    sys.exit(1)

# ── 시끄러운 MuPDF 경고 숨기기
fitz.TOOLS.mupdf_display_errors(False)

# ===== 전처리 유틸 (최적화) =====
@lru_cache(maxsize=10000)  # 반복되는 텍스트 정규화 캐시
def normalize_unicode_cached(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\r", "\n")

def normalize_unicode(text: str) -> str:
    return normalize_unicode_cached(text) if text else ""

# 정규식 컴파일 (성능 향상)
HYPHEN_PATTERN = re.compile(r"-\n(?=[A-Za-z0-9])")
SPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINE_PATTERN = re.compile(r"\n{3,}")

def fix_hyphen_linebreaks(text: str) -> str:
    """줄 끝 하이픈으로 끊긴 영어 단어 복원"""
    return HYPHEN_PATTERN.sub("", text)

def collapse_spaces(text: str) -> str:
    """공백과 줄바꿈 정리"""
    text = SPACE_PATTERN.sub(" ", text)
    text = NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()

def remove_simple_headers_footers(pages: List[str]) -> List[str]:
    """페이지마다 반복되는 헤더/푸터 제거 (최적화)"""
    if len(pages) < 2:  # 페이지 수가 적으면 스킵
        return pages
    
    def get_first_last_lines(page_text):
        lines = page_text.splitlines()
        first = next((line.strip() for line in lines if line.strip()), "")
        last = next((line.strip() for line in reversed(lines) if line.strip()), "")
        return first, last

    # 첫/마지막 줄 추출
    first_last_pairs = [get_first_last_lines(p) for p in pages]
    firsts, lasts = zip(*first_last_pairs)
    
    # 가장 빈번한 첫/마지막 줄 찾기 (최소 30% 이상 반복되는 것만)
    min_occurrences = max(2, len(pages) * 0.3)
    
    from collections import Counter
    first_counts = Counter(firsts)
    last_counts = Counter(lasts)
    
    common_first = next((text for text, count in first_counts.most_common() 
                        if count >= min_occurrences and len(text) > 5), "")
    common_last = next((text for text, count in last_counts.most_common() 
                       if count >= min_occurrences and len(text) > 5), "")

    # 헤더/푸터 제거
    cleaned = []
    for page in pages:
        lines = page.splitlines()
        if lines and common_first and lines[0].strip() == common_first:
            lines = lines[1:]
        if lines and common_last and lines[-1].strip() == common_last:
            lines = lines[:-1]
        cleaned.append("\n".join(lines))
    
    return cleaned

def clean_text_per_doc(pages: List[str], strip_headers: bool = False) -> str:
    """문서 전체 텍스트 정리"""
    # 빈 페이지 필터링
    pages = [normalize_unicode(page) for page in pages if page and page.strip()]
    
    if not pages:
        return ""
    
    if strip_headers and len(pages) > 1:
        pages = remove_simple_headers_footers(pages)
    
    text = "\n\n".join(pages)
    text = fix_hyphen_linebreaks(text)
    text = collapse_spaces(text)
    return text

# ===== 문장 분리/청크 (최적화) =====
@lru_cache(maxsize=1000)
def split_sentences_ko_cached(text_hash: int, text: str) -> Tuple[str, ...]:
    """문장 분리 결과 캐시"""
    sents = [s.strip() for s in kss.split_sentences(text) if s and len(s.strip()) >= 3]
    return tuple(sents)

def split_sentences_ko(text: str) -> List[str]:
    """한국어 최적화 문장 경계 분리"""
    if not text or len(text) < 10:
        return []
    
    text_hash = hash(text)
    cached_result = split_sentences_ko_cached(text_hash, text)
    return list(cached_result)

def chunk_by_sentences(sentences: List[str], target_len: int = 800, overlap_sents: int = 2) -> List[str]:
    """문장 기반 청킹 (최적화)"""
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # 현재 청크에 추가할 수 있는지 확인
        if current_length + sentence_len + 1 <= target_len or not current_chunk:
            current_chunk.append(sentence)
            current_length += sentence_len + 1
        else:
            # 현재 청크 완성
            chunks.append(" ".join(current_chunk))
            
            # 새 청크 시작 (겹침 문장 포함)
            overlap_start = max(0, len(current_chunk) - overlap_sents)
            current_chunk = current_chunk[overlap_start:] + [sentence]
            current_length = sum(len(s) + 1 for s in current_chunk)
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ===== PDF 텍스트 추출 (최적화) =====
def extract_pages_words(pdf_path: str, max_pages: Optional[int] = None, 
                       line_merge_tol: float = 2.0) -> List[str]:
    """단어 단위로 추출해 줄별 정렬·공백 복원"""
    pages = []
    
    try:
        with fitz.open(pdf_path) as doc:
            page_range = range(min(len(doc), max_pages) if max_pages else len(doc))
            
            for page_num in page_range:
                page = doc[page_num]
                words = page.get_text("words")
                
                if not words:
                    pages.append("")
                    continue
                
                # 좌표 기준 정렬 (y좌표 우선, 그 다음 x좌표)
                words.sort(key=lambda w: (round(w[1], 1), w[0]))
                
                # 줄별로 그룹화
                lines = []
                current_line = []
                current_y = None
                
                for word_data in words:
                    x0, y0, x1, y1, word = word_data[:5]
                    
                    if current_y is None or abs(y0 - current_y) < line_merge_tol:
                        current_line.append(word)
                        current_y = y0 if current_y is None else current_y
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]
                        current_y = y0
                
                # 마지막 줄 처리
                if current_line:
                    lines.append(" ".join(current_line))
                
                pages.append("\n".join(lines))
                
    except Exception as e:
        raise Exception(f"PDF 읽기 실패: {e}")
    
    return pages

def extract_pages_text(pdf_path: str, max_pages: Optional[int] = None) -> List[str]:
    """기본 텍스트 추출"""
    pages = []
    
    try:
        with fitz.open(pdf_path) as doc:
            page_range = range(min(len(doc), max_pages) if max_pages else len(doc))
            
            for page_num in page_range:
                page_text = doc[page_num].get_text() or ""
                pages.append(page_text)
                
    except Exception as e:
        raise Exception(f"PDF 읽기 실패: {e}")
    
    return pages

# ===== 병렬 PDF 처리 =====
def process_single_pdf(pdf_path: str, args) -> Optional[Dict]:
    """단일 PDF 처리 (병렬 처리용)"""
    doc_id = os.path.basename(pdf_path)
    
    try:
        # PDF 텍스트 추출
        if args.extract_mode == "words":
            pages = extract_pages_words(pdf_path, max_pages=args.max_pages, 
                                      line_merge_tol=args.line_merge_tol)
            # words 결과가 비면 text로 폴백
            if not any(p.strip() for p in pages):
                pages = extract_pages_text(pdf_path, max_pages=args.max_pages)
        else:
            pages = extract_pages_text(pdf_path, max_pages=args.max_pages)
        
        # 전처리
        cleaned = clean_text_per_doc(pages, strip_headers=args.strip_headers)
        
        if len(cleaned) < args.min_doc_len:
            return {"error": f"텍스트가 너무 적음({len(cleaned)} chars)", "doc_id": doc_id}
        
        # 문장 분리 및 청킹
        sentences = split_sentences_ko(cleaned)
        chunks = chunk_by_sentences(sentences, target_len=args.chunk_size, 
                                  overlap_sents=args.overlap_sents)
        
        if not chunks:
            return {"error": "청크 0개", "doc_id": doc_id}
        
        return {
            "doc_id": doc_id,
            "path": pdf_path,
            "cleaned_text": cleaned,
            "chunks": chunks,
            "num_sentences": len(sentences),
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "doc_id": doc_id}

# ===== 입력 목록 =====
def load_pdf_list(list_file: Path, explicit_list: Optional[List[str]]) -> List[str]:
    """PDF 목록 로드"""
    if explicit_list:
        return [str(Path(p).resolve()) for p in explicit_list]
    
    if not list_file.exists():
        print(f"[오류] {list_file} 가 없습니다. 먼저 scan_pdfs.py로 목록 파일을 만들어주세요.")
        sys.exit(1)
    
    paths = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(line)
    
    if not paths:
        print(f"[오류] {list_file} 안에 경로가 없습니다.")
        sys.exit(1)
    
    return paths

# ===== 메인 =====
def main():
    start_time = time()
    
    ap = argparse.ArgumentParser(description="PDF 전처리 + 문장 청크 + 임베딩 (최적화 버전)")
    ap.add_argument("--list-file", default="top5_pdfs.txt", help="PDF 절대경로 목록 파일")
    ap.add_argument("--pdf", nargs="*", help="목록 파일 대신 직접 PDF 절대경로 지정")
    ap.add_argument("--out", default="artifacts", help="결과 저장 폴더")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer 모델")
    ap.add_argument("--chunk-size", type=int, default=800, help="청크 크기")
    ap.add_argument("--overlap-sents", type=int, default=2, help="문장 겹침 개수")
    ap.add_argument("--min-doc-len", type=int, default=500, help="최소 문서 길이")
    ap.add_argument("--batch-size", type=int, default=64, help="임베딩 배치 크기")
    ap.add_argument("--extract-mode", choices=["words", "text"], default="words", help="PDF 텍스트 추출 방식")
    ap.add_argument("--max-pages", type=int, default=None, help="문서당 최대 처리 페이지 수")
    ap.add_argument("--line-merge-tol", type=float, default=2.0, help="words 모드 줄 병합 tolerance")
    ap.add_argument("--strip-headers", action="store_true", help="반복 헤더/푸터 제거")
    ap.add_argument("--workers", type=int, default=2, help="병렬 처리 스레드 수")
    ap.add_argument("--save-texts", action="store_true", help="전처리된 텍스트 파일도 저장")
    args = ap.parse_args()

    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.save_texts:
        (OUT_DIR / "texts").mkdir(exist_ok=True)

    pdf_list = load_pdf_list(Path(args.list_file), args.pdf)
    print(f"[정보] 처리 대상 {len(pdf_list)}개")
    for i, p in enumerate(pdf_list, 1):
        print(f" {i}. {os.path.basename(p)}")

    # 모델 로드
    print(f"\n[모델] {args.model} 로딩중...")
    model = SentenceTransformer(args.model)

    # 병렬 PDF 처리
    print(f"\n[처리] {args.workers}개 스레드로 PDF 처리중...")
    processed_results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf, args): pdf for pdf in pdf_list}
        
        for i, future in enumerate(as_completed(future_to_pdf), 1):
            pdf_path = future_to_pdf[future]
            doc_name = os.path.basename(pdf_path)
            
            try:
                result = future.result()
                if result.get("success"):
                    processed_results.append(result)
                    print(f"  ✓ [{i}/{len(pdf_list)}] {doc_name}: {len(result['chunks'])} chunks")
                else:
                    print(f"  ✗ [{i}/{len(pdf_list)}] {doc_name}: {result.get('error', '알 수 없는 오류')}")
            except Exception as e:
                print(f"  ✗ [{i}/{len(pdf_list)}] {doc_name}: 처리 실패 - {e}")

    if not processed_results:
        print("[오류] 처리된 PDF가 없습니다.")
        sys.exit(1)

    print(f"\n성공적으로 처리된 PDF: {len(processed_results)}개")

    # 임베딩 생성
    all_chunks = []
    all_records = []
    all_metas = []

    for result in processed_results:
        doc_id = result["doc_id"]
        chunks = result["chunks"]
        
        # 텍스트 파일 저장 (옵션)
        if args.save_texts:
            with open(OUT_DIR / "texts" / f"{doc_id}.txt", "w", encoding="utf-8") as f:
                f.write(result["cleaned_text"])
        
        # 청크와 메타데이터 수집
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_records.append({
                "doc_id": doc_id,
                "chunk_id": j,
                "text": chunk,
                "source": result["path"]
            })
        
        all_metas.append({
            "doc_id": doc_id,
            "path": result["path"],
            "num_chunks": len(chunks),
            "num_sentences": result["num_sentences"],
            "num_chars": len(result["cleaned_text"])
        })

    # 임베딩 생성
    print(f"\n[임베딩] 총 {len(all_chunks)}개 청크 임베딩 생성중...")
    embeddings = model.encode(all_chunks, convert_to_numpy=True, 
                            show_progress_bar=True, batch_size=args.batch_size)

    # 결과 저장
    print("\n[저장] 결과 파일 생성중...")
    
    # 청크 데이터 저장
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 임베딩 저장
    np.save(OUT_DIR / "embeddings.npy", embeddings.astype(np.float32))
    
    # 메타데이터 저장
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(all_metas, f, ensure_ascii=False, indent=2)

    # 메모리 정리
    del embeddings, all_chunks
    gc.collect()

    elapsed = time() - start_time
    print(f"\n✅ 완료! (총 소요시간: {elapsed:.1f}초)")
    print(f" - {OUT_DIR / 'chunks.jsonl'} ({len(all_records):,}개 청크)")
    print(f" - {OUT_DIR / 'embeddings.npy'} ({len(all_records):,}개 벡터)")
    print(f" - {OUT_DIR / 'meta.json'}")
    if args.save_texts:
        print(f" - {OUT_DIR / 'texts'}/*.txt")

    # 처리 통계
    total_chunks = sum(meta["num_chunks"] for meta in all_metas)
    total_chars = sum(meta["num_chars"] for meta in all_metas)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
    
    print(f"\n📊 처리 통계:")
    print(f" - 문서: {len(all_metas)}개")
    print(f" - 총 청크: {total_chunks:,}개")
    print(f" - 총 문자: {total_chars:,}개")
    print(f" - 평균 청크 크기: {avg_chunk_size:.0f}자")
    print(f" - 처리 속도: {total_chunks/elapsed:.1f} 청크/초")

if __name__ == "__main__":
    main()