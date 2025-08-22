import torch.nn as nn
import torch.nn.functional as F
import torch as T

class SolitaireNetwork1(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        n_classes: int, 
        embedding_dim: int = 16,
        hidden_dims: int = 64,
        num_heads: int = 4,
        **kwargs):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        
        self.card_embeddings = nn.Embedding(num_embeddings=52 + 2, embedding_dim=embedding_dim)
        
        self._initialize_tableau_layers()
        self._initialize_foundation_layers()
        self._initialize_waste_layers()
        self._initalize_stock_layers()
        
    def _initialize_tableau_layers(self):
        self.tableau_flatten = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.embedding_dim,
                self.hidden_dims,
                kernel_size=(15, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dims,
                self.hidden_dims,
                1
            ),
        )
        self.tableau_to_tableau_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dims,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dims*4,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.tableau_to_foundation_mha = (
            nn.MultiheadAttention(
                embed_dim=self.hidden_dims,
                num_heads=self.num_heads,
                kdim=self.hidden_dims,
                vdim=self.hidden_dims,
                batch_first=True
            )
        )
        
        self.tableau_to_foundation_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims // 2),
            nn.ReLU()
        )
        self.tableau_output_layer = nn.Linear(self.hidden_dims * 3 // 2, 10)   
    
    def _initialize_foundation_layers(self):
        
        self.foundation_flatten = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                self.embedding_dim,
                self.hidden_dims,
                kernel_size=4
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.hidden_dims,
                self.hidden_dims,
                1
            )
        )
        
        self.foundation_to_tableau_mha = (
            nn.MultiheadAttention(
                embed_dim=self.hidden_dims,
                num_heads=self.num_heads,
                kdim=self.hidden_dims,
                vdim=self.hidden_dims,
                batch_first=True
            )
        )
        
        self.foundation_output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, 10)
        )
        
    def _initalize_stock_layers(self):
        self.stock_output_layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        
    def _initialize_waste_layers(self):
        self.waste_prepare = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        
        self.waste_to_tableau_mha = (
            nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads // 2,
                kdim=self.hidden_dims,
                vdim=self.hidden_dims,
                batch_first=True
            )
        )
        self.waste_to_tableau_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU()
        )
        
        self.waste_to_foundation_mha = (
            nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                kdim=self.hidden_dims,
                vdim=self.hidden_dims,
                batch_first=True
            )
        )
        self.waste_to_foundation_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU()
        )
        
        self.waste_output_layer = nn.Linear(2 * self.embedding_dim, 10)
        
    def _forward_preprocess_layers(self, input_tensor):
        input_tensor = [
            input_x.unsqueeze(0) if input_x.dim() == dim_x else input_x
            for input_x, dim_x in zip(input_tensor, [2, 1, 1, 1])
        ]
        input_tableau, input_foundation, input_waste, input_stock = input_tensor
        input_stock = input_stock.to(T.float32)
        
        input_tableau = self.card_embeddings(input_tableau)
        input_foundation = self.card_embeddings(input_foundation)
        input_waste = self.card_embeddings(input_waste)
        
        input_tableau = input_tableau.permute(0, 3, 1, 2)
        input_foundation = input_foundation.permute(0, 2, 1)
        
        tableau_x = self.tableau_flatten(input_tableau)
        tableau_x = tableau_x.squeeze(2).permute(0, 2, 1)
        
        foundation_x = self.foundation_flatten(input_foundation)
        foundation_x = foundation_x.permute(0, 2, 1)
        
        stock_x = input_stock
        waste_x = self.waste_prepare(input_waste)
        
        return tableau_x, foundation_x, stock_x, waste_x
    
    def _forward_tableau(self, tableau_x, foundation_x):
        tableau_to_tableau = self.tableau_to_tableau_transformer(tableau_x)
        
        tableau_to_foundation, _ = self.tableau_to_foundation_mha(tableau_x, foundation_x, foundation_x)
        tableau_to_foundation = self.tableau_to_foundation_layers(tableau_to_foundation)
        
        tableau_concat = T.concat([tableau_to_tableau, tableau_to_foundation], dim=-1)
        tableau_output = self.tableau_output_layer(tableau_concat)
        
        return tableau_output
        
    def _forward_foundation(self, foundation_x, tableau_x):
        foundation_to_tableau, _ = self.foundation_to_tableau_mha(foundation_x, tableau_x, tableau_x)
        
        foundation_output = self.foundation_output_layers(foundation_to_tableau)
        return foundation_output
    
    def _forward_stock(self, stock_x):
        stock_output = self.stock_output_layers(stock_x)
        return stock_output.unsqueeze(1)
    
    def _forward_waste(self, waste_x, tableau_x, foundation_x):
        waste_to_tableau, _ = self.waste_to_tableau_mha(waste_x, tableau_x, tableau_x)
        waste_to_tableau = self.waste_to_tableau_layers(waste_to_tableau)
        
        waste_to_foundation, _ = self.waste_to_foundation_mha(waste_x, foundation_x, foundation_x)
        waste_to_foundation = self.waste_to_foundation_layers(waste_to_foundation)
        
        waste_concat = T.concat([waste_to_tableau, waste_to_foundation], axis=-1)
        
        waste_output = self.waste_output_layer(waste_concat)
        return waste_output
        
    def forward(self, input_tensor: tuple[T.Tensor], temperature: float = 1.):
        tableau_x, foundation_x, stock_x, waste_x = self._forward_preprocess_layers(input_tensor)
        
        tableau_output = self._forward_tableau(tableau_x, foundation_x)
        foundation_output = self._forward_foundation(foundation_x, tableau_x)
        stock_output = self._forward_stock(stock_x)
        waste_output = self._forward_waste(waste_x, tableau_x, foundation_x)

        logits = T.concat([
            tableau_output,
            stock_output,
            waste_output,
            foundation_output
        ], dim=1)
        logits = logits.flatten(start_dim=1)
        
        return logits, F.softmax(logits / temperature, dim=-1)
