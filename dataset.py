import numpy as np
import torch
from torch.utils.data import Dataset


class RatingWithNegDataset(Dataset):
    """
    Training dataset: real ratings + negative sampling.
    IDs start from 0.
    """
    def __init__(self, train_df, n_items, num_neg=4, seed=42):
        self.n_items = n_items
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

        self.pos_users = train_df["user_id"].values.astype(np.int64)
        self.pos_items = train_df["item_id"].values.astype(np.int64)
        self.pos_ratings = train_df["rating"].values.astype(np.float32)
        self.n_pos = len(self.pos_users)

        self.user_pos_items = {}
        for u, i in zip(self.pos_users, self.pos_items):
            self.user_pos_items.setdefault(int(u), set()).add(int(i))

        self.users = None
        self.items = None
        self.ratings = None
        self.resample_negatives()

    def resample_negatives(self):
        n_total = self.n_pos * (1 + self.num_neg)
        users = np.zeros(n_total, dtype=np.int64)
        items = np.zeros(n_total, dtype=np.int64)
        ratings = np.zeros(n_total, dtype=np.float32)

        idx = 0
        for k in range(self.n_pos):
            u = int(self.pos_users[k])
            seen = self.user_pos_items[u]

            users[idx] = u
            items[idx] = int(self.pos_items[k])
            ratings[idx] = self.pos_ratings[k]
            idx += 1

            for _ in range(self.num_neg):
                neg_i = self.rng.integers(0, self.n_items)  # 0-indexed
                while neg_i in seen:
                    neg_i = self.rng.integers(0, self.n_items)
                users[idx] = u
                items[idx] = neg_i
                ratings[idx] = 0.0
                idx += 1

        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )


class SasRecTrainDataset(Dataset):
    """
    SASRec pretraining dataset. IDs start from 0. Padding value = -1.
    """
    def __init__(self, user_history, n_users, n_items, max_len=50, seed=42):
        self.user_history = user_history
        self.n_users = n_users
        self.n_items = n_items
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.padding_value = -1

        self.valid_users = [u for u in range(n_users)
                            if len(user_history.get(u, [])) >= 2]

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        u = self.valid_users[idx]
        seq_items = self.user_history[u]

        seq = np.full(self.max_len, self.padding_value, dtype=np.int64)
        pos = np.full(self.max_len, self.padding_value, dtype=np.int64)
        neg = np.full(self.max_len, self.padding_value, dtype=np.int64)

        nxt = seq_items[-1]
        ptr = self.max_len - 1
        seen = set(seq_items)

        for item in reversed(seq_items[:-1]):
            seq[ptr] = item
            pos[ptr] = nxt

            neg_item = self.rng.integers(0, self.n_items)  # 0-indexed
            while neg_item in seen:
                neg_item = self.rng.integers(0, self.n_items)
            neg[ptr] = neg_item

            nxt = item
            ptr -= 1
            if ptr < 0:
                break

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )