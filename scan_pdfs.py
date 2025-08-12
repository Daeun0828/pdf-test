# scan_pdfs.py
import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF

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

def main():
    parser = argparse.ArgumentParser(description="PDF 폴더 스캔 후 상위 N개 경로 저장")
    parser.add_argument("pdf_dir", help="PDF들이 들어있는 폴더 경로")
    parser.add_argument("--topk", type=int, default=5, help="상위 몇 개를 저장할지 (기본 5)")
    parser.add_argument("--out", default="top5_pdfs.txt", help="경로 저장 파일명 (기본 top5_pdfs.txt)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"경로가 존재하지 않습니다: {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdfs:
        print("PDF 파일이 없습니다.")
        sys.exit(0)

    rows = []  # (name, abs_path, pages, text_pages, chars)
    for p in pdfs:
        pages, text_pages, chars = analyze_pdf(p)
        rows.append((p.name, str(p.resolve()), pages, text_pages, chars))

    # 텍스트 페이지/문자 수 많은 순으로 정렬
    rows.sort(key=lambda x: (x[3], x[4]), reverse=True)

    # 표 출력 (상위 50개 미리보기)
    print(f"{'name':60} | pages | text_pages | chars | path")
    print("-" * 120)
    for r in rows[:50]:
        print(f"{r[0]:60} | {r[2]:5} | {r[3]:10} | {r[4]:6} | {r[1]}")

    # 상위 N개 중에서 '정상 파일'만 골라 저장
    top_paths = [r[1] for r in rows if r[2] != -1 and r[3] > 0][:args.topk]
    with open(args.out, "w", encoding="utf-8") as f:
        for path in top_paths:
            f.write(path + "\n")

    print(f"\n상위 {len(top_paths)}개 절대경로를 {args.out}에 저장했습니다.")

if __name__ == "__main__":
    main()
