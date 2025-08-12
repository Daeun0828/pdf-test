# search.py - 개선된 의미론적 검색
import sys, json, argparse, re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import time

# 설정
ARTIFACTS_DIR = Path("./artifacts")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SemanticSearch:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR, model_name=MODEL_NAME):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.chunks = None
        self.doc_stats = None
        
    def load_data(self):
        """데이터 로드 및 검증"""
        print(f"📂 {self.artifacts_dir}에서 데이터 로딩 중...")
        
        # 필수 파일 존재 확인
        required_files = ["embeddings.npy", "chunks.jsonl"]
        for file in required_files:
            file_path = self.artifacts_dir / file
            if not file_path.exists():
                print(f"❌ {file_path}가 없습니다. preprocess_embed.py를 먼저 실행하세요.")
                sys.exit(1)
        
        # 임베딩 로드
        try:
            self.embeddings = np.load(self.artifacts_dir / "embeddings.npy", allow_pickle=False)
            if self.embeddings.size == 0 or self.embeddings.ndim != 2:
                raise ValueError("잘못된 임베딩 차원")
            print(f"  ✓ 임베딩: {self.embeddings.shape}")
        except Exception as e:
            print(f"❌ 임베딩 로드 실패: {e}")
            sys.exit(1)
        
        # 청크 데이터 로드
        try:
            self.chunks = []
            with open(self.artifacts_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line.strip())
                        self.chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  라인 {line_no} JSON 오류: {e}")
            
            print(f"  ✓ 청크: {len(self.chunks)}개")
            
            if len(self.chunks) != len(self.embeddings):
                print(f"⚠️  청크 개수({len(self.chunks)})와 임베딩 개수({len(self.embeddings)})가 다릅니다.")
                
        except Exception as e:
            print(f"❌ 청크 데이터 로드 실패: {e}")
            sys.exit(1)
        
        # 문서 통계 계산
        self._calculate_doc_stats()
        
        print("✅ 데이터 로드 완료!\n")
    
    def _calculate_doc_stats(self):
        """문서별 통계 계산"""
        self.doc_stats = defaultdict(lambda: {"chunk_count": 0, "total_chars": 0})
        
        for chunk in self.chunks:
            doc_id = chunk["doc_id"]
            self.doc_stats[doc_id]["chunk_count"] += 1
            self.doc_stats[doc_id]["total_chars"] += len(chunk["text"])
    
    def load_model(self):
        """모델 로드"""
        if self.model is None:
            print(f"🤖 {self.model_name} 모델 로딩 중...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ 모델 로드 완료!")
    
    def cosine_similarity(self, query_vec, embeddings):
        """코사인 유사도 계산 (최적화)"""
        # L2 정규화
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        
        # 내적으로 코사인 유사도 계산
        return embeddings_norm @ query_norm
    
    def search(self, query, top_k=10, threshold=0.0, group_by_doc=False):
        """의미론적 검색 수행"""
        start_time = time.time()
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # 유사도 계산
        similarities = self.cosine_similarity(query_embedding, self.embeddings)
        
        # 결과 정렬
        ranked_indices = similarities.argsort()[::-1]
        
        results = []
        shown_count = 0
        
        if group_by_doc:
            # 문서별로 그룹화해서 보여주기
            doc_results = defaultdict(list)
            
            for idx in ranked_indices:
                if similarities[idx] < threshold:
                    break
                
                chunk = self.chunks[idx]
                doc_id = chunk["doc_id"]
                
                doc_results[doc_id].append({
                    "chunk": chunk,
                    "score": similarities[idx],
                    "index": idx
                })
                
                if len(doc_results) >= top_k:
                    break
            
            # 문서별 최고 점수로 정렬
            for doc_id, doc_chunks in sorted(doc_results.items(), 
                                           key=lambda x: max(c["score"] for c in x[1]), 
                                           reverse=True):
                results.append({
                    "doc_id": doc_id,
                    "chunks": sorted(doc_chunks, key=lambda x: x["score"], reverse=True),
                    "max_score": max(c["score"] for c in doc_chunks)
                })
        else:
            # 개별 청크별로 보여주기
            for idx in ranked_indices:
                if similarities[idx] < threshold:
                    break
                
                chunk = self.chunks[idx]
                results.append({
                    "chunk": chunk,
                    "score": similarities[idx],
                    "index": idx
                })
                
                shown_count += 1
                if shown_count >= top_k:
                    break
        
        search_time = time.time() - start_time
        
        return {
            "results": results,
            "query": query,
            "total_time": search_time,
            "total_chunks": len(self.chunks),
            "shown_count": len(results)
        }
    
    def print_stats(self):
        """데이터 통계 출력"""
        if not self.doc_stats:
            return
            
        print("📊 문서별 통계:")
        total_chunks = sum(stats["chunk_count"] for stats in self.doc_stats.values())
        total_chars = sum(stats["total_chars"] for stats in self.doc_stats.values())
        
        for doc_id, stats in sorted(self.doc_stats.items(), 
                                  key=lambda x: x[1]["chunk_count"], reverse=True):
            chunk_pct = stats["chunk_count"] / total_chunks * 100
            print(f"  📄 {doc_id}")
            print(f"     청크: {stats['chunk_count']:,}개 ({chunk_pct:.1f}%)")
            print(f"     문자: {stats['total_chars']:,}개")
        
        print(f"\n총 {len(self.doc_stats)}개 문서, {total_chunks:,}개 청크, {total_chars:,}개 문자\n")

def pretty_preview(text, max_len=500):
    """텍스트 미리보기 (자연스러운 자르기)"""
    text = text.replace("\n", " ").strip()
    
    if len(text) <= max_len:
        return text
    
    # 문장 경계에서 자르기
    truncated = text[:max_len]
    
    # 한국어/영어 문장 끝 찾기
    sentence_endings = ['. ', '! ', '? ', '다. ', '요. ', '니다. ', '습니다. ']
    
    best_cut = -1
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > best_cut:
            best_cut = pos + len(ending)
    
    if best_cut > max_len * 0.7:  # 70% 이상 위치에서 끝나면 사용
        return truncated[:best_cut].strip()
    
    # 문장 끝을 못 찾으면 공백에서 자르기
    space_pos = truncated.rfind(' ')
    if space_pos > max_len * 0.8:
        return truncated[:space_pos] + "..."
    
    return truncated + "..."

def print_search_results(search_results, group_by_doc=False):
    """검색 결과 출력"""
    results = search_results["results"]
    query = search_results["query"]
    total_time = search_results["total_time"]
    
    print(f"🔍 질의: '{query}'")
    print(f"⏱️  검색 시간: {total_time:.3f}초")
    print(f"📝 결과: {len(results)}개\n")
    
    if not results:
        print("❌ 검색 결과가 없습니다. 다른 키워드를 시도해보세요.\n")
        return
    
    if group_by_doc:
        # 문서별 그룹 출력
        for i, doc_result in enumerate(results, 1):
            doc_id = doc_result["doc_id"]
            chunks = doc_result["chunks"][:3]  # 문서당 최대 3개 청크만
            max_score = doc_result["max_score"]
            
            print(f"📄 [{i}] {doc_id} (최고점수: {max_score:.3f})")
            
            for j, chunk_data in enumerate(chunks):
                chunk = chunk_data["chunk"]
                score = chunk_data["score"]
                print(f"  └─ 청크 {chunk['chunk_id']} (점수: {score:.3f})")
                print(f"     {pretty_preview(chunk['text'], 400)}")
                if j < len(chunks) - 1:
                    print()
            print("\n" + "─"*80 + "\n")
    else:
        # 개별 청크 출력
        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            score = result["score"]
            
            print(f"📄 [{i}] {chunk['doc_id']} (청크 {chunk['chunk_id']}) - 점수: {score:.3f}")
            print(f"   {pretty_preview(chunk['text'])}")
            print()

def main():
    ap = argparse.ArgumentParser(description="의미론적 문서 검색 (개선 버전)")
    ap.add_argument("query", nargs="?", help="검색할 질의 (없으면 대화형 모드)")
    ap.add_argument("--artifacts", default="./artifacts", help="artifacts 폴더 경로")
    ap.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer 모델")
    ap.add_argument("--topk", type=int, default=10, help="상위 k개 결과")
    ap.add_argument("--threshold", type=float, default=0.0, help="최소 유사도 점수")
    ap.add_argument("--group-by-doc", action="store_true", help="문서별로 그룹화")
    ap.add_argument("--stats", action="store_true", help="데이터 통계만 출력")
    args = ap.parse_args()
    
    # 검색 엔진 초기화
    search_engine = SemanticSearch(args.artifacts, args.model)
    search_engine.load_data()
    
    if args.stats:
        search_engine.print_stats()
        return
    
    search_engine.load_model()
    search_engine.print_stats()
    
    if args.query:
        # 단일 검색
        results = search_engine.search(
            args.query, 
            top_k=args.topk, 
            threshold=args.threshold,
            group_by_doc=args.group_by_doc
        )
        print_search_results(results, args.group_by_doc)
    else:
        # 대화형 모드
        print("🔍 대화형 검색 모드 (종료: 'quit' 또는 Ctrl+C)")
        print("💡 명령어: stats (통계), clear (화면 지우기)")
        print("=" * 60)
        
        try:
            while True:
                query = input("\n질의: ").strip()
                
                if not query or query.lower() in ["quit", "exit", "q"]:
                    break
                
                if query.lower() == "stats":
                    search_engine.print_stats()
                    continue
                
                if query.lower() == "clear":
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                results = search_engine.search(
                    query, 
                    top_k=args.topk, 
                    threshold=args.threshold,
                    group_by_doc=args.group_by_doc
                )
                print_search_results(results, args.group_by_doc)
                
        except KeyboardInterrupt:
            print("\n👋 검색을 종료합니다.")

if __name__ == "__main__":
    main()