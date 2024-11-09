import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, hidden_state):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.book_embeddings = nn.Embedding(num_books, embedding_dim)
        self.linear_embedding = nn.Linear(hidden_state, embedding_dim)
        self.output = nn.Linear(embedding_dim, 6)

    def forward(self, user, book, tag_embedding):
        user_embedding = self.user_embeddings(user)
        book_embedding = self.book_embeddings(book)
        tag_embedding_proj = self.linear_embedding(tag_embedding)
        book_intergrate = book_embedding + tag_embedding_proj
        return (user_embedding * book_intergrate).sum(dim = 1)