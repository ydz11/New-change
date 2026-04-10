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


def build_eval_candidates(eval_user_item_pairs, n_items, user_seen_items, num_neg=100, seed=42):
    rng = np.random.default_rng(seed)
    users = eval_user_item_pairs[:, 0].astype(np.int64)
    pos_items = eval_user_item_pairs[:, 1].astype(np.int64)

    candidates = np.zeros((len(users), 1 + num_neg), dtype=np.int64)
    candidates[:, 0] = pos_items

    all_items = np.arange(1, n_items + 1, dtype=np.int64)

    for idx, (u, pos_i) in enumerate(zip(users.tolist(), pos_items.tolist())):
        # Exclude items the user has seen in prior splits (same criteria as
        # training negative sampling), plus the current positive item itself
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
def evaluate_model(model, eval_users, eval_candidates, k, device, batch_size=4096):
    model.eval()

    ds = EvalDataset(eval_users, eval_candidates)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Collect per-row results with user IDs for per-user averaging
    row_users = []
    row_hits = []
    row_ndcgs = []

    for u, items in dl:
        bsz, n_cand = items.shape
        u_ids = u.clone()  # keep original user IDs for grouping
        u_exp = u.to(device).view(-1, 1).expand(bsz, n_cand).reshape(-1)
        items = items.to(device).reshape(-1)

        scores = model(u_exp, items).view(bsz, n_cand)
        pos_score = scores[:, 0].unsqueeze(1)
        rank = 1 + (scores[:, 1:] > pos_score).sum(dim=1)

        for uid, r in zip(u_ids.tolist(), rank.cpu().tolist()):
            row_users.append(uid)
            row_hits.append(1.0 if r <= k else 0.0)
            row_ndcgs.append(ndcg_from_rank(int(r), k))

    # Average per-user, then average across users
    from collections import defaultdict
    user_hits = defaultdict(list)
    user_ndcgs = defaultdict(list)
    for uid, h, n in zip(row_users, row_hits, row_ndcgs):
        user_hits[uid].append(h)
        user_ndcgs[uid].append(n)

    avg_hr = np.mean([np.mean(v) for v in user_hits.values()])
    avg_ndcg = np.mean([np.mean(v) for v in user_ndcgs.values()])

    return float(avg_hr), float(avg_ndcg)