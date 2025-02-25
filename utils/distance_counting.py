import torch

class DistanceCounting:
    """Class for computing similarity between face embeddings."""
    @staticmethod
    def cosine_similarity(emb1, emb2):
        """Computes cosine similarity between two embeddings."""
        emb1 = emb1 / emb1.norm()
        emb2 = emb2 / emb2.norm()
        return torch.dot(emb1.squeeze(), emb2.squeeze()).item()

    @staticmethod
    def euclidean_distance(emb1, emb2):
        """Computes Euclidean distance between two embeddings."""
        return torch.norm(emb1 - emb2).item()
