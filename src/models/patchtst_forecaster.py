import torch
import torch.nn as nn

class PatchTSTForecaster(nn.Module):
    def __init__(self, input_dim, horizon=1, patch_len=8, stride=4,
                 d_model=64, n_heads=4, n_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride

        # Flatten each patch across all features
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        # Pad to make sequence divisible by stride (optional)
        pad_len = (self.stride - (seq_len % self.stride)) % self.stride
        if pad_len:
            x = torch.cat([x, torch.zeros(batch, pad_len, self.input_dim, device=x.device)], dim=1)
            seq_len += pad_len

        # Extract patches using unfold
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: (batch, num_patches, input_dim, patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.view(batch, -1, self.patch_len * self.input_dim)

        patch_embeds = self.patch_embed(patches)          # (batch, num_patches, d_model)
        encoded = self.transformer(patch_embeds)          # (batch, num_patches, d_model)
        pooled = encoded.mean(dim=1)                      # (batch, d_model)
        out = self.head(pooled)                           # (batch, horizon)
        return out
