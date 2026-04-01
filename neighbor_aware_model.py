import torch
import torch.nn as nn


class NeighborAware(nn.Module):
    """
    Proposed model: neighbor embedding concatenation + MLP rating regression.
    IDs start from 0. Padding value for missing neighbors = -1.
    """

    def __init__(self, user_emb, item_emb, user_neighbors, item_neighbors,
                 n_users, n_items, k=5, freeze_pretrained=False, dropout=0.2):
        super().__init__()
        self.k = k
        emb_dim = user_emb.shape[1]

        # Load pretrained SASRec embeddings (no padding_idx since IDs start from 0)
        self.user_emb = nn.Embedding.from_pretrained(user_emb, freeze=freeze_pretrained)
        self.item_emb = nn.Embedding.from_pretrained(item_emb, freeze=freeze_pretrained)

        # Build neighbor lookup tables. Padding = -1 (no valid neighbor).
        user_topk = torch.full((n_users, k), -1, dtype=torch.long)
        for u, neigh_list in user_neighbors.items():
            neigh_list = neigh_list[:k]
            user_topk[u, :len(neigh_list)] = torch.tensor(neigh_list, dtype=torch.long)

        item_topk = torch.full((n_items, k), -1, dtype=torch.long)
        for i, neigh_list in item_neighbors.items():
            neigh_list = neigh_list[:k]
            item_topk[i, :len(neigh_list)] = torch.tensor(neigh_list, dtype=torch.long)

        self.register_buffer("user_topk_buf", user_topk)
        self.register_buffer("item_topk_buf", item_topk)

        # MLP
        mlp_input_dim = 2 * (k + 1) * emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 2, mlp_input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 4, 1),
        )

    def forward(self, user, item):
        batch_size = user.size(0)

        u_target = self.user_emb(user)
        i_target = self.item_emb(item)

        # Get neighbor IDs and embeddings
        u_nei_ids = self.user_topk_buf[user]       # [batch, k]
        i_nei_ids = self.item_topk_buf[item]       # [batch, k]

        # Replace -1 padding with 0 for embedding lookup (0 is a valid ID but we'll zero it out)
        u_nei_safe = u_nei_ids.clamp(min=0)
        i_nei_safe = i_nei_ids.clamp(min=0)

        u_nei_emb = self.user_emb(u_nei_safe)      # [batch, k, emb_dim]
        i_nei_emb = self.item_emb(i_nei_safe)      # [batch, k, emb_dim]

        # Zero out padding positions (where original ID was -1)
        u_pad_mask = (u_nei_ids == -1).unsqueeze(-1)
        i_pad_mask = (i_nei_ids == -1).unsqueeze(-1)
        u_nei_emb = u_nei_emb.masked_fill(u_pad_mask, 0.0)
        i_nei_emb = i_nei_emb.masked_fill(i_pad_mask, 0.0)

        # Flatten and concatenate
        u_nei_flat = u_nei_emb.view(batch_size, -1)
        i_nei_flat = i_nei_emb.view(batch_size, -1)

        user_side = torch.cat([u_target, u_nei_flat], dim=-1)
        item_side = torch.cat([i_target, i_nei_flat], dim=-1)

        mlp_input = torch.cat([user_side, item_side], dim=-1)
        output = self.mlp(mlp_input)

        return output.squeeze(-1)