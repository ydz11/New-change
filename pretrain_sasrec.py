import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class SimpleSASRec(nn.Module):
    def __init__(self, n_items, hidden_units=64, max_len=50, num_blocks=2, num_heads=1, dropout_rate=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden_units = hidden_units
        self.max_len = max_len

        # Item embedding: n_items entries (IDs 0..n_items-1), no padding_idx needed
        # since ID=0 is now a valid item
        self.item_embedding = nn.Embedding(n_items, hidden_units)

        # Positional embedding: position 0..max_len-1 in the sequence
        self.positional_embedding = nn.Embedding(max_len, hidden_units)

        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def encode(self, seq, padding_value=-1):
        """
        seq: [B, L] item IDs. padding_value indicates empty positions.
        """
        device = seq.device
        positions = torch.arange(seq.size(1), device=device).unsqueeze(0).expand_as(seq)

        # For embedding lookup, replace padding with 0 (any valid index)
        seq_for_emb = seq.clone()
        seq_for_emb[seq_for_emb == padding_value] = 0

        x = self.item_embedding(seq_for_emb) + self.positional_embedding(positions)
        x = self.dropout(x)

        # Mask: True where padding
        padding_mask = (seq == padding_value)

        # Zero out padding positions in the embedding
        x = x * (~padding_mask).float().unsqueeze(-1)

        L = seq.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        h = self.layer_norm(h)
        return h

    def forward(self, seq, pos_target, neg_target, padding_value=-1):
        """
        Standard SASRec BPR training loss.
        seq:        [B, L] input sequence
        pos_target: [B, L] correct next item at each position
        neg_target: [B, L] random wrong item at each position
        """
        h = self.encode(seq, padding_value)

        # Replace padding in targets with 0 for embedding lookup
        pos_for_emb = pos_target.clone()
        pos_for_emb[pos_for_emb == padding_value] = 0
        neg_for_emb = neg_target.clone()
        neg_for_emb[neg_for_emb == padding_value] = 0

        pos_emb = self.item_embedding(pos_for_emb)
        neg_emb = self.item_embedding(neg_for_emb)

        pos_logits = (h * pos_emb).sum(dim=-1)
        neg_logits = (h * neg_emb).sum(dim=-1)

        mask = (pos_target != padding_value).float()

        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_logits) + 1e-24) * mask

        loss = (pos_loss + neg_loss).sum() / (mask.sum() + 1e-24)
        return loss

    @torch.no_grad()
    def export_user_item_embeddings(self, user_history, n_users, device, padding_value=-1):
        self.eval()
        user_emb = torch.zeros((n_users, self.hidden_units), dtype=torch.float32, device=device)

        for u in range(n_users):
            seq_items = user_history.get(u, [])
            if len(seq_items) == 0:
                continue

            seq = np.full(self.max_len, padding_value, dtype=np.int64)
            trunc = seq_items[-self.max_len:]
            seq[-len(trunc):] = trunc

            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            h = self.encode(seq_tensor, padding_value)

            valid_pos = (seq_tensor != padding_value).sum().item() - 1
            if valid_pos >= 0:
                user_emb[u] = h[0, valid_pos]

        item_emb = self.item_embedding.weight.detach().clone()
        return user_emb.cpu(), item_emb.cpu()


def pretrain_sasrec(
        train_dataset,
        user_history,
        n_users,
        n_items,
        device,
        hidden_units=64,
        max_len=50,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.2,
        batch_size=128,
        lr=1e-3,
        epochs=20,
):
    model = SimpleSASRec(
        n_items=n_items,
        hidden_units=hidden_units,
        max_len=max_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for _, seq, pos_target, neg_target in loader:
            seq = seq.to(device)
            pos_target = pos_target.to(device)
            neg_target = neg_target.to(device)

            loss = model(seq, pos_target, neg_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[SASRec pretrain] epoch={epoch:02d}, loss={avg_loss:.4f}")

    user_emb, item_emb = model.export_user_item_embeddings(user_history, n_users, device)
    return user_emb, item_emb