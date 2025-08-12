# export_texts.py - 개선된 텍스트 추출기
import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

def slugify(name: str) -> str:
    """파일명을 안전하게 변환"""
    # 확장자 분리
    if '.' in name:
        stem = name[:name.rfind('.')]
        ext = name[name.rfind('.'):]
    else:
        stem, ext = name, ""
    
    # 특수문자 제거/변환
    safe_name = re.sub(r"[\\/:\*\?\"<>\|]", "_", stem)
    safe_name = re.sub(r"\s+", "_", safe_name)  # 공백을 언더스코어로
    safe_name = re.sub(r"_+", "_", safe_name)   # 연속 언더스코어 제거
    safe_name = safe_name.strip("._")           # 앞뒤 특수문자 제거
    
    # 너무 긴 이름 줄이기
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    
    return safe_name if safe_name else "document"

def load_chunks(chunks_path: Path) -> Dict[str, List[Tuple]]:
    """청크 데이터를 문서별로 그룹화"""
    by_doc = defaultdict(list)
    
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    doc_id = obj.get("doc_id", f"unknown_{line_no}")
                    chunk_id = obj.get("chunk_id", 0)
                    text = obj.get("text", "")
                    source = obj.get("source")
                    
                    by_doc[doc_id].append((chunk_id, text, source))
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  라인 {line_no} JSON 오류: {e}")
                    continue
    
    except Exception as e:
        raise Exception(f"청크 파일 읽기 실패: {e}")
    
    # 청크 ID 순서로 정렬
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda x: x[0])
    
    return dict(by_doc)

def merge_chunks(chunks: List[Tuple], mode: str = "standard") -> str:
    """청크들을 병합 (여러 모드 지원)"""
    if not chunks:
        return ""
    
    texts = [text for _, text, _ in chunks]
    
    if mode == "standard":
        # 기본: 두 줄 공백으로 구분
        return "\n\n".join(texts)
    
    elif mode == "compact":
        # 압축: 한 줄 공백으로 구분
        return "\n".join(texts)
    
    elif mode == "seamless":
        # 끊김없이: 공백 하나로 이어붙임
        return " ".join(text.strip() for text in texts)
    
    elif mode == "chapter":
        # 챕터별: 번호 매기기
        result = []
        for i, text in enumerate(texts, 1):
            result.append(f"## 청크 {i}\n\n{text}")
        return "\n\n".join(result)
    
    else:
        return "\n\n".join(texts)

def write_txt(out_dir: Path, doc_id: str, merged_text: str, stats: Dict) -> Path:
    """TXT 파일 작성"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(doc_id)}.txt"
    
    # 간단한 헤더 추가
    header = f"문서: {doc_id}\n"
    header += f"청크 수: {stats['chunk_count']}개\n"
    header += f"총 문자: {stats['total_chars']:,}자\n"
    header += f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += "=" * 50 + "\n\n"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + merged_text)
    
    return out_path

def write_md(out_dir: Path, doc_id: str, merged_text: str, source_path: Optional[str], 
            stats: Dict, mode: str = "standard") -> Path:
    """마크다운 파일 작성"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(doc_id)}.md"
    
    # 마크다운 헤더 생성
    header = f"# {doc_id}\n\n"
    
    # 메타데이터 테이블
    header += "## 문서 정보\n\n"
    header += "| 항목 | 값 |\n"
    header += "|------|----|\n"
    if source_path:
        header += f"| 원본 경로 | `{source_path}` |\n"
    header += f"| 청크 수 | {stats['chunk_count']}개 |\n"
    header += f"| 총 문자 | {stats['total_chars']:,}자 |\n"
    header += f"| 평균 청크 크기 | {stats['avg_chunk_size']:.0f}자 |\n"
    header += f"| 생성 시간 | {time.strftime('%Y-%m-%d %H:%M:%S')} |\n\n"
    
    # 목차 생성 (chapter 모드일 때)
    if mode == "chapter":
        header += "## 목차\n\n"
        chunk_count = stats['chunk_count']
        for i in range(1, chunk_count + 1):
            header += f"- [청크 {i}](#청크-{i})\n"
        header += "\n"
    
    header += "---\n\n## 내용\n\n"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + merged_text)
    
    return out_path

def calculate_stats(chunks: List[Tuple]) -> Dict:
    """청크 통계 계산"""
    if not chunks:
        return {"chunk_count": 0, "total_chars": 0, "avg_chunk_size": 0}
    
    total_chars = sum(len(text) for _, text, _ in chunks)
    chunk_count = len(chunks)
    avg_chunk_size = total_chars / chunk_count if chunk_count > 0 else 0
    
    return {
        "chunk_count": chunk_count,
        "total_chars": total_chars,
        "avg_chunk_size": avg_chunk_size
    }

def print_summary(doc_stats: Dict[str, Dict]):
    """전체 요약 출력"""
    if not doc_stats:
        return
    
    print("\n" + "="*60)
    print("📊 처리 요약")
    print("="*60)
    
    total_docs = len(doc_stats)
    total_chunks = sum(stats["chunk_count"] for stats in doc_stats.values())
    total_chars = sum(stats["total_chars"] for stats in doc_stats.values())
    
    print(f"총 문서: {total_docs}개")
    print(f"총 청크: {total_chunks:,}개")
    print(f"총 문자: {total_chars:,}개")
    
    if total_docs > 0:
        avg_chunks_per_doc = total_chunks / total_docs
        avg_chars_per_doc = total_chars / total_docs
        print(f"문서당 평균 청크: {avg_chunks_per_doc:.1f}개")
        print(f"문서당 평균 문자: {avg_chars_per_doc:,.0f}자")
    
    print("\n문서별 상세:")
    for doc_id, stats in sorted(doc_stats.items(), 
                               key=lambda x: x[1]["total_chars"], reverse=True):
        print(f"  📄 {doc_id}")
        print(f"     청크: {stats['chunk_count']}개, 문자: {stats['total_chars']:,}자")

def main():
    parser = argparse.ArgumentParser(
        description="chunks.jsonl을 문서별로 병합하여 텍스트 파일 생성 (개선 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
병합 모드:
  standard  - 기본 모드 (청크간 두 줄 공백)
  compact   - 압축 모드 (청크간 한 줄 공백) 
  seamless  - 연속 모드 (공백 하나로 이어붙임)
  chapter   - 챕터 모드 (번호 매기기)

사용 예시:
  python export_texts.py
  python export_texts.py --mode compact --out-md markdown
  python export_texts.py --filter "백신,GMP" --stats-only
        """
    )
    
    parser.add_argument("--artifacts", default="artifacts", 
                       help="artifacts 폴더 경로 (기본: artifacts)")
    parser.add_argument("--out-txt", default="artifacts/texts_merged", 
                       help="TXT 출력 폴더")
    parser.add_argument("--out-md", default=None, 
                       help="MD(마크다운) 출력 폴더 (선택)")
    parser.add_argument("--mode", choices=["standard", "compact", "seamless", "chapter"],
                       default="standard", help="병합 모드")
    parser.add_argument("--filter", default=None,
                       help="특정 문서만 처리 (쉼표로 구분된 키워드)")
    parser.add_argument("--exclude", default=None,
                       help="제외할 문서 (쉼표로 구분된 키워드)")
    parser.add_argument("--min-chunks", type=int, default=1,
                       help="최소 청크 수 (기본: 1)")
    parser.add_argument("--stats-only", action="store_true",
                       help="통계만 출력하고 파일 생성 안 함")
    
    args = parser.parse_args()

    # 입력 파일 확인
    art = Path(args.artifacts)
    chunks_path = art / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"❌ {chunks_path} 파일이 없습니다.")
        print("먼저 preprocess_embed.py를 실행하세요.")
        return 1

    print(f"📂 {chunks_path}에서 청크 데이터 로딩 중...")
    by_doc = load_chunks(chunks_path)
    
    if not by_doc:
        print("❌ 처리할 문서가 없습니다.")
        return 1
    
    print(f"✅ {len(by_doc)}개 문서 발견")
    
    # 필터링
    if args.filter:
        filter_keywords = [k.strip().lower() for k in args.filter.split(",")]
        filtered = {}
        for doc_id, chunks in by_doc.items():
            if any(keyword in doc_id.lower() for keyword in filter_keywords):
                filtered[doc_id] = chunks
        by_doc = filtered
        print(f"🔍 필터 적용: {len(by_doc)}개 문서 선택")
    
    if args.exclude:
        exclude_keywords = [k.strip().lower() for k in args.exclude.split(",")]
        filtered = {}
        for doc_id, chunks in by_doc.items():
            if not any(keyword in doc_id.lower() for keyword in exclude_keywords):
                filtered[doc_id] = chunks
        by_doc = filtered
        print(f"🚫 제외 적용: {len(by_doc)}개 문서 남음")
    
    # 최소 청크 수 필터
    if args.min_chunks > 1:
        filtered = {doc_id: chunks for doc_id, chunks in by_doc.items() 
                   if len(chunks) >= args.min_chunks}
        by_doc = filtered
        print(f"📊 최소 청크 필터: {len(by_doc)}개 문서 남음")
    
    # 통계 수집
    doc_stats = {}
    for doc_id, chunks in by_doc.items():
        doc_stats[doc_id] = calculate_stats(chunks)
    
    # 통계만 출력하고 종료
    if args.stats_only:
        print_summary(doc_stats)
        return 0
    
    # 파일 생성
    txt_dir = Path(args.out_txt)
    md_dir = Path(args.out_md) if args.out_md else None
    
    print(f"\n📝 {args.mode} 모드로 파일 생성 중...")
    
    processed = 0
    for doc_id, chunks in by_doc.items():
        try:
            merged_text = merge_chunks(chunks, args.mode)
            stats = doc_stats[doc_id]
            
            # TXT 파일 생성
            txt_path = write_txt(txt_dir, doc_id, merged_text, stats)
            
            # MD 파일 생성 (옵션)
            md_path = None
            if md_dir:
                source_path = chunks[0][2] if chunks and chunks[0][2] else None
                md_path = write_md(md_dir, doc_id, merged_text, source_path, stats, args.mode)
            
            # 진행 상황 출력
            print(f"  ✅ {doc_id}")
            print(f"      청크: {stats['chunk_count']}개 → 문자: {stats['total_chars']:,}자")
            print(f"      📁 {txt_path.name}")
            if md_path:
                print(f"      📁 {md_path.name}")
            
            processed += 1
            
        except Exception as e:
            print(f"  ❌ {doc_id}: 처리 실패 - {e}")
    
    # 최종 요약
    print(f"\n✅ 완료: {processed}개 문서 처리")
    print(f"📁 TXT: {txt_dir.resolve()}")
    if md_dir:
        print(f"📁 MD:  {md_dir.resolve()}")
    
    print_summary(doc_stats)
    
    return 0

if __name__ == "__main__":
    exit(main())