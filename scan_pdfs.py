# scan_pdfs.py
import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
from time import time

# 시끄러운 MuPDF 경고 숨기기
fitz.TOOLS.mupdf_display_errors(False)

def analyze_pdf(path: Path):
    """
    반환: (pages, text_pages, chars)
    - pages: 총 페이지 수 (열기 실패 시 -1)
    - text_pages: 텍스트가 감지된 페이지 수(간단 기준)
    - chars: 전체 텍스트 문자 수
    """
    try:
        with fitz.open(path) as doc:
            pages = len(doc)
            text_pages = 0
            chars = 0
            for page in doc:
                t = page.get_text() or ""
                chars += len(t)
                if len(t.strip()) > 100:  # 텍스트가 '있다'고 볼 간단 임계값
                    text_pages += 1
            return pages, text_pages, chars
    except Exception:
        return -1, -1, -1  # 깨짐/암호 등

def get_top5_with_progress(pdfs, max_workers=4):
    """메모리 효율적인 상위 5개 선별 + 진행률 표시"""
    total = len(pdfs)
    processed = 0
    heap = []  # 최소 힙으로 상위 5개만 유지
    failed_count = 0
    
    print("PDF 분석 중...")
    start_time = time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_pdf = {executor.submit(analyze_pdf, pdf): pdf for pdf in pdfs}
        
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            processed += 1
            
            try:
                pages, text_pages, chars = future.result()
                
                if pages == -1:  # 실패한 파일
                    failed_count += 1
                else:
                    # 힙을 사용해 상위 5개만 유지 (메모리 절약)
                    score = (text_pages, chars)  # 정렬 기준
                    item = (score, pdf_path.name, str(pdf_path.resolve()), pages, text_pages, chars)
                    
                    if len(heap) < 5:
                        heapq.heappush(heap, item)
                    elif score > heap[0][0]:  # 현재 최소값보다 큰 경우만
                        heapq.heappushpop(heap, item)
            
            except Exception:
                failed_count += 1
            
            # 진행률 표시 (10개마다 또는 마지막)
            if processed % 10 == 0 or processed == total:
                elapsed = time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                
                print(f"\r진행: {processed}/{total} ({processed/total*100:.1f}%) "
                      f"| 속도: {rate:.1f}개/초 | 남은시간: {remaining:.0f}초 "
                      f"| 실패: {failed_count}개", end="")
    
    print()  # 줄바꿈
    elapsed = time() - start_time
    print(f"완료! 총 소요시간: {elapsed:.1f}초 (평균 {rate:.2f}개/초)")
    
    # 힙을 리스트로 변환하고 내림차순 정렬
    results = sorted(heap, key=lambda x: x[0], reverse=True)
    return results, failed_count

def main():
    parser = argparse.ArgumentParser(description="PDF 폴더 스캔 후 상위 5개 경로 저장 (고속 처리)")
    parser.add_argument("pdf_dir", help="PDF들이 들어있는 폴더 경로")
    parser.add_argument("--out", default="top5_pdfs.txt", help="경로 저장 파일명 (기본 top5_pdfs.txt)")
    parser.add_argument("--workers", type=int, default=4, help="병렬 처리 스레드 수 (기본 4)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"경로가 존재하지 않습니다: {pdf_dir}")
        sys.exit(1)

    # 대소문자 구분 없이 PDF 파일 찾기
    pdfs = []
    for pattern in ["**/*.pdf", "**/*.PDF"]:
        pdfs.extend(pdf_dir.glob(pattern))
    
    pdfs = sorted(set(pdfs))  # 중복 제거
    
    if not pdfs:
        print("PDF 파일이 없습니다.")
        print(f"확인된 폴더: {pdf_dir.resolve()}")
        print("하위 폴더 내용:")
        for item in pdf_dir.rglob("*"):
            if item.is_file():
                print(f"  {item.suffix}: {item.name}")
        sys.exit(0)

    print(f"총 {len(pdfs)}개 PDF 파일 발견")
    
    # 고속 처리로 상위 5개 선별
    top_results, failed_count = get_top5_with_progress(pdfs, args.workers)
    
    if failed_count > 0:
        print(f"⚠️  {failed_count}개 파일 처리 실패 (손상/암호화 등)")

    # 결과 표시
    print(f"\n{'순위'} {'name':50} | pages | text_pages | chars")
    print("-" * 80)
    
    for i, (score, name, abs_path, pages, text_pages, chars) in enumerate(top_results, 1):
        print(f"{i:2}★ {name:50} | {pages:5} | {text_pages:10} | {chars:6}")

    # 상위 5개 경로 저장
    top5_paths = [result[2] for result in top_results]  # abs_path만 추출
    
    with open(args.out, "w", encoding="utf-8") as f:
        for path in top5_paths:
            f.write(path + "\n")

    print(f"\n★ 상위 {len(top5_paths)}개 PDF 경로를 {args.out}에 저장했습니다:")
    for i, (_, name, _, _, text_pages, chars) in enumerate(top_results, 1):
        print(f"  {i}. {name} (텍스트 페이지: {text_pages}, 문자수: {chars:,})")

if __name__ == "__main__":
    main()