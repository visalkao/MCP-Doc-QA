import faiss
import numpy as np
import os
import pickle

class FaissStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vectors: np.ndarray, metadatas: list):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5):
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, k)
        results = []
        for row_idx, ids in enumerate(I):
            row = []
            for id_ in ids:
                if id_ < len(self.metadatas):
                    row.append((self.metadatas[id_], float(D[row_idx][list(ids).index(id_)])))
            results.append(row)
        return results
    
    def get_all(self):
        """
        Return all stored snippets with a dummy score = 1.0
        """
        return [(meta, 1.0) for meta in self.metadatas]

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        meta_path = self.index_path + '.meta'
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            meta_path = self.index_path + '.meta'
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.metadatas = pickle.load(f)
