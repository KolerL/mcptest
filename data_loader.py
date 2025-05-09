import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class LocalKnowledge:
    # 初始化类时加载Excel数据、预训练模型，并预计算所有问题的嵌入向量
    def __init__(self, excel_path):
        self.df = pd.read_excel(excel_path)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self._precompute_embeddings()
    
    # 将知识库中的每个问题转换为向量并存储，加速后续搜索
    def _precompute_embeddings(self):
        """预计算所有问题的向量"""
        self.question_embeddings = self.model.encode(
            self.df['question'].tolist(),
            convert_to_numpy=True
        )
    
    def search(self, query, threshold=0.7, top_k=3):
        """语义搜索最相关问题"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                results.append((idx, score))
        
        # 按相似度排序并取前top_k
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {
                "question": self.df.iloc[idx]['question'],
                "answer": self.df.iloc[idx]['answer'],
                "score": float(score)
            }
            for idx, score in sorted_results
        ]