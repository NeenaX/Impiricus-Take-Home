import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CourseRAGPipeline:
    def __init__(
        self, 
        model_name: str = "BAAI/bge-base-en-v1.5",
        csv_path: str = "../data/courses.csv",
        index_path: str = "embeddings/faiss_index.bin",
        metadata_path: str = "embeddings/metadata.pkl"
    ):
        self.model_name = model_name
        self.csv_path = csv_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Load data in 
        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna('') # Handles NaNs

        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def prepare_text(self, row: pd.Series):
        parts = []

        # Course code and title
        if row['course_code']:
            parts.append(f"Course: {row['course_code']}")
        if row['title']:
            parts.append(f"Title: {row['title']}")
        
        # Description
        if row['description']:
            parts.append(f"Description: {row['description']}")
        
        # Department
        if row['department']:
            parts.append(f"Department: {row['department']}")
        
        # Instructor,meeting times, and prerequisites (from CAB courses)
        if row.get('instructor', ''):
            parts.append(f"Instructor: {row['instructor']}")
        if row.get('meeting_times', ''):
            parts.append(f"Meeting Times: {row['meeting_times']}")
        if row.get('prerequisites', ''):
            parts.append(f"Prerequisites: {row['prerequisites']}")
        
        return " | ".join(parts)

    def build_index(self, rebuild: bool=False):
        """
        Builds or loads FAISS index

        Args:
            rebuild (bool, optional): True to rebuild the index even if it exists Defaults to False.
        """
        if not rebuild:
            try:
                self.load_index()
                print("Loaded existing FAISS index")
                return
            except Exception as e:
                print("No existing index found, building new one...")
        
        # Prepare text for embeddings
        texts = self.df.apply(self.prepare_text, axis=1).to_list()

        # Generating embeddings
        print("Generating embeddings...")
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Creating FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # IndexFlatIP for cosine similarity search
        self.index.add(self.embeddings.astype('float32'))

        # Build TFIDF for hybrid search
        print("Building TFIDF index... ")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def save_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, self.index_path)
        
        metadata = {
            'embeddings': self.embeddings,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'model_name': self.model_name
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_index(self):
        """Load in FAISS index and metadata"""
        self.faiss_index = faiss.read_index(self.index_path)
        
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.embeddings = metadata['embeddings']
        self.tfidf_vectorizer = metadata['tfidf_vectorizer']
        self.tfidf_matrix = metadata['tfidf_matrix']
    
    def filter_by_metadata(
        self, 
        department: Optional[str] = None,
        source: Optional[str] = None,
        instructor: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create boolean mask for metadata filtering
        
        Args:
            department: Filter by department code
            source: Filter by source ("CAB" or "Bulletin")
            instructor: Filter by instructor name (partial match)
            
        Returns:
            Boolean numpy array indicating which courses match filters
        """
        mask = np.ones(len(self.df), dtype=bool)

        if department:
            mask &= (self.df['department'].str.upper() == department.upper()).values
        
        if source:
            mask &= (self.df['source'].str.upper() == source.upper()).values
        
        if instructor:
            mask &= self.df['instructor'].str.contains(
                instructor, case=False, na=False, regex=False
            ).values
        
        return mask
    
    def semantic_search(
        self, 
        query: str, 
        k: int = 10,
        filter_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Perform semantic vector search using FAISS
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_mask: Boolean mask for pre-filtering results
            
        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')
        
        if filter_mask is None:
            # Search all documents
            distances, indices = self.faiss_index.search(query_embedding, k)
        else:
            # Apply filter by searching only filtered embeddings
            filtered_indices = np.where(filter_mask)[0]
            
            if len(filtered_indices) == 0:
                return [], []
            
            # Create temporary index with filtered embeddings
            filtered_embeddings = self.embeddings[filtered_indices]
            temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
            temp_index.add(filtered_embeddings.astype('float32'))
            
            # Search and map back to original indices
            k_actual = min(k, len(filtered_indices))
            distances, temp_indices = temp_index.search(query_embedding, k_actual)
            indices = np.array([[filtered_indices[i] for i in temp_indices[0]]])
        
        return indices[0].tolist(), distances[0].tolist()
    
    def keyword_search(
        self, 
        query: str, 
        k: int = 10,
        filter_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Perform keyword-based TFIDF search
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_mask: Boolean mask for pre-filtering results
            
        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Transform query using TFIDF
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Compute similarities
        if filter_mask is None:
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:
            filtered_matrix = self.tfidf_matrix[filter_mask]
            similarities_filtered = cosine_similarity(query_vec, filtered_matrix).flatten()
            
            # Create full similarity array
            similarities = np.zeros(len(self.df))
            filtered_indices = np.where(filter_mask)[0]
            similarities[filtered_indices] = similarities_filtered
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        department: Optional[str] = None,
        source: Optional[str] = None,
        instructor: Optional[str] = None,
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            query: Search query text
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            department: Optional department filter
            source: Optional source filter
            instructor: Optional instructor filter
            
        Returns:
            List of course dictionaries with scores and metadata
        """
        # Apply metadata filters if any
        filter_mask = self.filter_by_metadata(
            department=department,
            source=source,
            instructor=instructor,
        )
        
        # Perform both searches for the hybrid approach
        semantic_indices, semantic_scores = self.semantic_search(
            query, k=k*2, filter_mask=filter_mask
        )
        keyword_indices, keyword_scores = self.keyword_search(
            query, k=k*2, filter_mask=filter_mask
        )
        
        # Normalize scores to [0, 1]
        def normalize_scores(scores):
            if not scores or max(scores) == 0:
                return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        semantic_scores_norm = normalize_scores(semantic_scores)
        keyword_scores_norm = normalize_scores(keyword_scores)
        
        # Combine scores
        combined_scores = {}
        
        for idx, score in zip(semantic_indices, semantic_scores_norm):
            combined_scores[idx] = semantic_weight * score
        
        for idx, score in zip(keyword_indices, keyword_scores_norm):
            if idx in combined_scores:
                combined_scores[idx] += keyword_weight * score
            else:
                combined_scores[idx] = keyword_weight * score
        
        # Sort by combined score and get top k
        sorted_indices = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        # Format results
        results = []
        for idx, score in sorted_indices:
            course = self.df.iloc[idx].to_dict()
            course['similarity_score'] = float(score)
            course['index'] = int(idx)
            results.append(course)
        
        return results
    
    def retrieve_context(
        self, 
        query: str, 
        k: int = 5,
        **kwargs
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant courses and format context for LLM
        
        Args:
            query: Search query
            k: Number of courses to retrieve
            **kwargs: Additional filters (department, source, instructor)
            
        Returns:
            Tuple of (formatted_context_string, list_of_retrieved_courses)
        """
        # Perform hybrid search
        results = self.hybrid_search(query, k=k, **kwargs)
        
        # Format context for LLM
        context_parts = [f"Query: {query}\n"]
        context_parts.append(f"Retrieved {len(results)} relevant courses:\n")
        
        for i, course in enumerate(results, 1):
            context_parts.append(f"\n--- Course {i} (Similarity: {course['similarity_score']:.3f}) ---")
            context_parts.append(f"Code: {course['course_code']}")
            context_parts.append(f"Title: {course['title']}")
            context_parts.append(f"Department: {course['department']}")
            
            if course.get('instructor'):
                context_parts.append(f"Instructor: {course['instructor']}")
            if course.get('meeting_times'):
                context_parts.append(f"Meeting Times: {course['meeting_times']}")
            if course.get('prerequisites'):
                context_parts.append(f"Prerequisites: {course['prerequisites']}")
            
            context_parts.append(f"Description: {course['description'][:300]}...")
            context_parts.append(f"Source: {course['source']}")
        
        context = "\n".join(context_parts)
        return context, results


if __name__ == "__main__":
    # Initialize pipeline
    
    rag = CourseRAGPipeline(
        csv_path="../data/courses.csv",
        model_name="BAAI/bge-base-en-v1.5"
    )
    
    # # Build index (first time only)
    # rag.build_index(rebuild=False)
    # rag.save_index()
    rag.load_index()
    
    # Test queries
    print("\n" + "="*80)
    print("Test Query 1: Who teaches AFRI 0090?")
    print("="*80)
    context, results = rag.retrieve_context(
        "AFRI 0090 instructor",
        k=3
    )
    for result in results:
        print(f"\n{result['course_code']}: {result['title']}")
        print(f"  Score: {result['similarity_score']:.3f}")
        if result.get('instructor'):
            print(f"  Instructor: {result['instructor']}")
    
    print("\n" + "="*80)
    print("Test Query 2: Philosophy courses on metaphysics")
    print("="*80)
    context, results = rag.retrieve_context(
        "metaphysics philosophy",
        k=5,
        department="PHIL"
    )
    for result in results:
        print(f"\n{result['course_code']}: {result['title']}")
        print(f"  Score: {result['similarity_score']:.3f}")
    
    print("\n" + "="*80)
    print("Test Query 3: Machine learning courses on Fridays after 3pm")
    print("="*80)
    context, results = rag.retrieve_context(
        "machine learning",
        k=5,
        source="CAB",
    )
    for result in results:
        print(f"\n{result['course_code']}: {result['title']}")
        print(f"  Score: {result['similarity_score']:.3f}")
        if result.get('meeting_times'):
            print(f"  Times: {result['meeting_times']}")