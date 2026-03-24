import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    load_ratings,
    leave_one_out_split,
    build_user_history,
    build_user_seen_items,
    build_train_uir,
    build_ui,
    get_num_users_items,
)
from dataset import RatingWithNegDataset, SasRecTrainDataset
from neighbor_retrieval import build_neighbor_dicts
from pretrain_sasrec import pretrain_sasrec
from mf_model import MF
from ncf_model import NCF
from sasrec_ncf import SASRecNCF
from neighbor_aware_model import NeighborAware
from evaluate import build_eval_candidates, evaluate_model


# =============================================================
# Configuration
# =============================================================

DATA_PATH = "ml-1m/ratings.dat"
OUTPUT_DIR = Path("outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding factors to test (same as NCF paper Section 4.1)
FACTORS = [8, 16, 32, 64]

# Neighbor settings (proposal Step 1)
NEIGHBOR_K = 5

# Training settings
TRAIN_BATCH_SIZE = 256
TRAIN_EPOCHS = 40
NUM_NEG_TRAIN = 4       # NCF paper uses 4 negatives per positive
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5

# SASRec pretraining settings (proposal Step 2)
SASREC_MAXLEN = 50
SASREC_EPOCHS = 20
SASREC_BATCH_SIZE = 128

# Evaluation settings (NCF paper: top-10, 100 negatives)
TOP_K = 10
NUM_NEG_EVAL = 100


# =============================================================
# Training function
# =============================================================

def train_rating_model(model, train_dataset, valid_users, valid_candidates,
                       model_name, lr=LR, weight_decay=WEIGHT_DECAY):
    """
    Train a model with MSE loss + negative sampling.
    Early stopping based on validation NDCG@K.

    Args:
        model: nn.Module with forward(user, item) -> predicted_rating
        train_dataset: RatingWithNegDataset (re-samples negatives each epoch)
        valid_users: np.array of user IDs for validation
        valid_candidates: np.array [n_valid, 101] of candidate item IDs
        model_name: str for logging
        lr: learning rate
        weight_decay: L2 regularization
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_ndcg = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        # Re-sample negatives each epoch for diversity
        train_dataset.resample_negatives()
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        # --- Train ---
        model.train()
        epoch_losses = []

        for user, item, rating in train_loader:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            rating = rating.to(DEVICE)

            pred = model(user, item)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # --- Validate ---
        hr, ndcg = evaluate_model(model, valid_users, valid_candidates, TOP_K, DEVICE)
        avg_loss = float(np.mean(epoch_losses))

        print(f"[{model_name}] epoch={epoch:02d}, "
              f"train_mse={avg_loss:.4f}, "
              f"valid_hr@{TOP_K}={hr:.4f}, "
              f"valid_ndcg@{TOP_K}={ndcg:.4f}")

        # --- Early stopping ---
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


# =============================================================
# Plotting
# =============================================================

def plot_results(results, save_dir):
    """Plot NDCG and HR comparison across factors."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    factors = sorted(results.keys())
    model_names = list(next(iter(results.values())).keys())

    # NDCG plot
    plt.figure(figsize=(8, 5))
    for name in model_names:
        y = [results[f][name]["ndcg"] for f in factors]
        plt.plot(factors, y, marker="o", label=name)
    plt.xlabel("Factor")
    plt.ylabel(f"NDCG@{TOP_K}")
    plt.title("NDCG Comparison across Factors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "ndcg_compare.png", dpi=200)
    plt.close()

    # HR plot
    plt.figure(figsize=(8, 5))
    for name in model_names:
        y = [results[f][name]["hr"] for f in factors]
        plt.plot(factors, y, marker="o", label=name)
    plt.xlabel("Factor")
    plt.ylabel(f"HR@{TOP_K}")
    plt.title("HR Comparison across Factors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "hr_compare.png", dpi=200)
    plt.close()


# =============================================================
# Main
# =============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. Load data and split
    # -------------------------------------------------------
    df = load_ratings(DATA_PATH)
    n_users, n_items = get_num_users_items(df)
    train_df, valid_df, test_df = leave_one_out_split(df)

    print(f"Dataset loaded: #users={n_users}, #items={n_items}")
    print(f"  #train={len(train_df)}, #valid={len(valid_df)}, #test={len(test_df)}")

    # Prepare data structures
    train_uir = build_train_uir(train_df)
    valid_ui = build_ui(valid_df)
    test_ui = build_ui(test_df)
    user_history = build_user_history(train_df)

    # Seen items for negative sampling during evaluation
    user_train_seen = build_user_seen_items(train_df, n_users)

    user_train_valid_seen = [set() for _ in range(n_users + 1)]
    for row in train_df.itertuples(index=False):
        user_train_valid_seen[int(row.user_id)].add(int(row.item_id))
    for row in valid_df.itertuples(index=False):
        user_train_valid_seen[int(row.user_id)].add(int(row.item_id))

    # -------------------------------------------------------
    # Step 1: Compute cosine-based neighbors
    # -------------------------------------------------------
    print("\n--- Step 1: Computing neighbors ---")
    user_neighbors, item_neighbors = build_neighbor_dicts(
        train_uir=train_uir,
        n_users=n_users,
        n_items=n_items,
        k=NEIGHBOR_K,
        sim_threshold=0.0,
    )

    # -------------------------------------------------------
    # Build evaluation candidates (done once, shared by all models)
    # -------------------------------------------------------
    valid_users, valid_candidates = build_eval_candidates(
        valid_ui, n_items, user_train_seen, num_neg=NUM_NEG_EVAL, seed=42
    )
    test_users, test_candidates = build_eval_candidates(
        test_ui, n_items, user_train_valid_seen, num_neg=NUM_NEG_EVAL, seed=42
    )

    # -------------------------------------------------------
    # Build training dataset (shared by all models)
    # -------------------------------------------------------
    train_dataset = RatingWithNegDataset(
        train_df=train_df,
        n_items=n_items,
        num_neg=NUM_NEG_TRAIN,
        seed=42,
    )

    # -------------------------------------------------------
    # Run experiments for each factor
    # -------------------------------------------------------
    all_results = {}

    for factor in FACTORS:
        print(f"\n{'='*60}")
        print(f"Factor = {factor}, embedding_dim = {factor}")
        print(f"{'='*60}")

        emb_dim = factor  # embedding_dim = factor directly (matching NCF paper)

        # MLP hidden layers: tower/halving pattern (NCF paper Section 3.3)
        # Input is emb_dim*2 (concat user+item), then halve
        mlp_layers = [emb_dim * 2, emb_dim, emb_dim // 2]

        # Dropout: larger factor -> more regularization needed
        dropout = 0.1
        if factor >= 32:
            dropout = 0.3
        elif factor >= 16:
            dropout = 0.2

        # ---------------------------------------------------
        # Step 2: Pretrain SASRec for this embedding dimension
        # ---------------------------------------------------
        print(f"\n--- Step 2: Pretraining SASRec (dim={emb_dim}) ---")

        sasrec_train_dataset = SasRecTrainDataset(
            user_history=user_history,
            n_users=n_users,
            n_items=n_items,
            max_len=SASREC_MAXLEN,
            seed=42,
        )

        user_emb, item_emb = pretrain_sasrec(
            train_dataset=sasrec_train_dataset,
            user_history=user_history,
            n_users=n_users,
            n_items=n_items,
            device=DEVICE,
            hidden_units=emb_dim,
            max_len=SASREC_MAXLEN,
            num_blocks=2,
            num_heads=1,
            dropout_rate=0.2,
            batch_size=SASREC_BATCH_SIZE,
            lr=1e-3,
            epochs=SASREC_EPOCHS,
        )

        # ---------------------------------------------------
        # Build all models for this factor
        # ---------------------------------------------------
        models = {
            # Baseline 1: standard MF (no neural, no neighbor)
            "MF": MF(n_users, n_items, embedding_dim=emb_dim),

            # Baseline 2: NCF (neural, no neighbor, random init)
            "NCF": NCF(n_users, n_items, embedding_dim=emb_dim, hidden_dims=mlp_layers),

            # Baseline 3: SASRec pretrained embeddings + MLP (no neighbor)
            "SASRec-NCF": SASRecNCF(user_emb, item_emb, hidden_dims=mlp_layers,
                                     freeze_pretrained=False),

            # Proposed: SASRec embeddings + neighbor concatenation + MLP
            "NeighborAware": NeighborAware(
                user_emb, item_emb, user_neighbors, item_neighbors,
                n_users, n_items, k=NEIGHBOR_K,
                freeze_pretrained=False, dropout=dropout,
            ),
        }

        # ---------------------------------------------------
        # Train all models
        # ---------------------------------------------------
        best_models = {}
        for name, model in models.items():
            print(f"\n--- Training {name} (factor={factor}) ---")
            best_models[name] = train_rating_model(
                model=model,
                train_dataset=train_dataset,
                valid_users=valid_users,
                valid_candidates=valid_candidates,
                model_name=f"{name}-f{factor}",
            )

        # ---------------------------------------------------
        # Test all models
        # ---------------------------------------------------
        factor_results = {}
        print(f"\n--- Test Results (factor={factor}) ---")
        for name, model in best_models.items():
            hr, ndcg = evaluate_model(model, test_users, test_candidates, TOP_K, DEVICE)
            factor_results[name] = {"hr": hr, "ndcg": ndcg}
            print(f"  {name:15s}  HR@{TOP_K}={hr:.4f}  NDCG@{TOP_K}={ndcg:.4f}")

        all_results[factor] = factor_results

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------
    with open(OUTPUT_DIR / "results_by_factor.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    plot_results(all_results, OUTPUT_DIR)

    print(f"\nResults saved to {OUTPUT_DIR / 'results_by_factor.json'}")
    print(f"Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()