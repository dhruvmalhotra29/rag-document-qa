import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.bm25 = None
        # Cross-Encoder for reranking
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def add_embeddings(self, embeddings, documents):

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.documents.extend(documents)

        # Build BM25 index
        tokenized_documents = [
            self._tokenize(self._get_text(doc))
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query_embedding,
        query,
        k=3,
        mode="hybrid"
    ):

        if not self.documents:
            return []

        if mode == "faiss":
            return self._search_faiss(
                query_embedding,
                k
            )

        elif mode == "hybrid":
            return self._search_hybrid(
                query_embedding,
                query,
                k
            )

        else:
            raise ValueError(
                "mode must be 'faiss' or 'hybrid'"
            )

    # -------------------------
    # FAISS ONLY
    # -------------------------

    def _search_faiss(self, query_embedding, k):

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        search_k = min(k, len(self.documents))

        distances, indexes = self.index.search(
            query_embedding,
            search_k
        )

        best_distance = distances[0][0]

        print("FAISS best distance:", best_distance)
        print("FAISS distances:", distances)
        print("FAISS indexes:", indexes)

        if best_distance > 1.7:
            return []

        results = []

        for idx in indexes[0]:

            if idx == -1:
                continue

            results.append(
                self.documents[idx]
            )

        return results

    # -------------------------
    # HYBRID SEARCH
    # -------------------------

    def _search_hybrid(
        self,
        query_embedding,
        query,
        k
    ):

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        # Retrieve more candidates
        search_k = min(
            k * 3,
            len(self.documents)
        )

        # -------------------------
        # FAISS
        # -------------------------

        distances, indexes = self.index.search(
            query_embedding,
            search_k
        )

        semantic_results = []

        for rank, idx in enumerate(indexes[0]):

            if idx == -1:
                continue

            semantic_results.append({
                "doc": self.documents[idx],
                "rank": rank
            })

        # -------------------------
        # BM25
        # -------------------------

        tokenized_query = self._tokenize(query)

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        bm25_indexes = np.argsort(
            bm25_scores
        )[::-1][:search_k]

        keyword_results = []

        for rank, idx in enumerate(bm25_indexes):

            keyword_results.append({
                "doc": self.documents[idx],
                "rank": rank
            })

        # -------------------------
        # RRF combination
        # -------------------------

        combined = {}

        for result in semantic_results:

            doc_id = id(result["doc"])

            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": result["doc"],
                    "score": 0
                }

            combined[doc_id]["score"] += (
                1 / (60 + result["rank"] + 1)
            )

        for result in keyword_results:

            doc_id = id(result["doc"])

            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": result["doc"],
                    "score": 0
                }

            combined[doc_id]["score"] += (
                1 / (60 + result["rank"] + 1)
            )

        # Sort using RRF score
        results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Keep more candidates for Cross-Encoder
        candidates = [
            result["doc"]
            for result in results[:k * 3]
        ]

        # --------------------------------
        # Cross-Encoder reranking
        # --------------------------------

        pairs = [
            [query, self._get_text(doc)]
            for doc in candidates
        ]

        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nCross-Encoder scores:")

        for doc, score in reranked:
            print(
                round(float(score), 4),
                self._get_text(doc)[:100]
            )

        # Return final top K
        return [
            doc
            for doc, score in reranked[:k]
        ]

    # -------------------------
    # TEXT EXTRACTION
    # -------------------------

    def _get_text(self, document):

        if isinstance(document, dict):
            return document.get("text", "")

        return str(document)

    # -------------------------
    # TOKENIZATION
    # -------------------------

    def _tokenize(self, text):

        return text.lower().split()

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.bm25 = None