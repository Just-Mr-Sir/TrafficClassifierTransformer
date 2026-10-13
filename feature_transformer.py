import torch
import torch.nn as nn

class FeatureTransformer(nn.Module):
    def __init__(self, embed_dim, head_count, encode_layer_count, in_channels, out_channels, dropout):
        super(FeatureTransformer, self).__init__()

        #learned embed feature mappings (non-shared projections)
        self.weights = nn.Parameter(torch.randn(in_channels, embed_dim))
        self.bias = nn.Parameter(torch.zeros(in_channels, embed_dim))

        #feature identities
        self.feature_emb = nn.Embedding(in_channels, embed_dim)

        #multiply by small number (e.g., 0.02) to keep initial weights small.
        self.class_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        self.normalize = nn.LayerNorm(embed_dim)

        #transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=head_count,
            dim_feedforward= 4 * embed_dim,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encode_layer_count
        )

        #self.head = nn.Linear(embed_dim, out_channels)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_channels)
        )


    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.bias.unsqueeze(0)

        x = x + self.feature_emb(torch.arange(x.size(1), device=x.device))

        x = self.normalize(x)

        cls = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = self.encoder(x)

        return self.head(x[:, 0])