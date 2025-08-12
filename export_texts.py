# export_texts.py
import argparse
import json
from pathlib import Path
import re

def slugify(name: str) -> str:
    # 파일명 안전하게 변환 (확장자는 유지하지 않음)
    s = re.sub(r"[\\/:\*\?\"<>\|]", "_", name)
    s = s.strip().strip(".")
    return s if s else "document"

def load_chunks(chunks_path: Path):
    by_doc = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            by_doc.setdefault(doc_id, []).append((obj["chunk_id"], obj["text"], obj.get("source")))
    # 청크 순서 정렬
    for k in by_doc:
        by_doc[k].sort(key=lambda x: x[0])
    return by_doc

def merge_chunks(chunks):
    """단순 병합(겹침은 두 줄 공백으로 이어 붙임)"""
    texts = [t for _, t, _ in chunks]
    return "\n\n".join(texts)

def write_txt(out_dir: Path, doc_id: str, merged_text: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(doc_id)}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(merged_text)
    return out_path

def write_md(out_dir: Path, doc_id: str, merged_text: str, source_path: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(doc_id)}.md"
    header = f"# {doc_id}\n\n"
    if source_path:
        header += f"- 원본 경로: `{source_path}`\n\n---\n\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + merged_text)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="chunks.jsonl을 문서별로 병합하여 텍스트 파일 생성")
    parser.add_argument("--artifacts", default="artifacts", help="artifacts 폴더 경로 (기본: artifacts)")
    parser.add_argument("--out-txt", default="artifacts/texts_merged", help="TXT 출력 폴더")
    parser.add_argument("--out-md", default=None, help="MD(마크다운) 출력 폴더 (선택)")
    args = parser.parse_args()

    art = Path(args.artifacts)
    chunks_path = art / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} 파일이 없습니다. 먼저 preprocess_embed.py를 실행하세요.")

    by_doc = load_chunks(chunks_path)

    txt_dir = Path(args.out_txt)
    md_dir = Path(args.out_md) if args.out_md else None

    total = 0
    for doc_id, arr in by_doc.items():
        merged = merge_chunks(arr)
        txt_path = write_txt(txt_dir, doc_id, merged)

        src = arr[0][2] if arr and arr[0][2] else None
        if md_dir:
            write_md(md_dir, doc_id, merged, src)

        print(f"✔ {doc_id} -> {txt_path}")
        total += 1

    print(f"\n완료: {total}개 문서를 병합했습니다.")
    print(f"- TXT: {txt_dir.resolve()}")
    if md_dir:
        print(f"- MD : {md_dir.resolve()}")

if __name__ == "__main__":
    main()
