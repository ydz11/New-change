import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EvalDataset(Dataset):
    def __init__(self, users, candidates):
        self.users = users.astype(np.int64)
        self.candidates = candidates.astype(np.int64)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.candidates[idx]


def build_eval_candidates(pos_ui, n_items, user_seen_items, num_neg=100, seed=42):
    """
    For each user:
      candidates = [1 positive] + [num_neg sampled negatives]
    """
    rng = np.random.default_rng(seed)
    users = pos_ui[:, 0].astype(np.int64)
    pos_items = pos_ui[:, 1].astype(np.int64)

    candidates = np.zeros((len(users), 1 + num_neg), dtype=np.int64)
    candidates[:, 0] = pos_items

    all_items = np.arange(1, n_items + 1, dtype=np.int64)

    for idx, (u, pos_i) in enumerate(zip(users.tolist(), pos_items.tolist())):
        excluded = set(user_seen_items[int(u)])
        excluded.add(int(pos_i))

        mask = np.ones(n_items, dtype=bool)
        excluded_idx = np.array(list(excluded), dtype=np.int64) - 1
        mask[excluded_idx] = False
        pool = all_items[mask]

        sampled = rng.choice(pool, size=num_neg, replace=False)
        candidates[idx, 1:] = sampled

    return users, candidates


def ndcg_from_rank(rank, k):
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


@torch.no_grad()
def evaluate_model(model, users, candidates, k, device, batch_size=4096):
    model.eval()

    ds = EvalDataset(users, candidates)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    hits = []
    ndcgs = []

    for u, items in dl:
        bsz, n_cand = items.shape
        u = u.to(device).view(-1, 1).expand(bsz, n_cand).reshape(-1)
        items = items.to(device).reshape(-1)

        scores = model(u, items).view(bsz, n_cand)
        pos_score = scores[:, 0].unsqueeze(1)
        rank = 1 + (scores[:, 1:] > pos_score).sum(dim=1)

        for r in rank.cpu().tolist():
            hits.append(1.0 if r <= k else 0.0)
            ndcgs.append(ndcg_from_rank(int(r), k))

    return float(np.mean(hits)), float(np.mean(ndcgs))
