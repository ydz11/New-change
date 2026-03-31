import pandas as pd
import numpy as np


def load_ratings(path):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="::", engine="python", names=cols)
    df = df.sort_values(["user_id", "timestamp"]).copy()
    return df


def filter_cold_start(df, min_user_interactions=5, min_item_interactions=5):
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)

        # 统计每个用户的交互数，过滤掉太少的
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        # 统计每个物品的交互数，过滤掉太少的
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)]

    df = df.reset_index(drop=True)
    return df


def reindex_ids(df):
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {old: new for new, old in enumerate(unique_users, start=1)}
    item_map = {old: new for new, old in enumerate(unique_items, start=1)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    return df


def get_num_users_items(df):
    n_users = int(df["user_id"].nunique())
    n_items = int(df["item_id"].nunique())
    return n_users, n_items


def ratio_split(df, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15):
    train_rows = []
    valid_rows = []
    test_rows = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        n = len(group)

        n_train = max(1, int(n * train_ratio))

        train_part = group.iloc[:n_train]
        remaining = group.iloc[n_train:]

        if len(remaining) == 0:
            train_rows.append(train_part)
            continue

        high_rating = remaining[remaining["rating"] >= 4]
        low_rating = remaining[remaining["rating"] < 4]

        if len(high_rating) < 2:
            train_rows.append(group)
            continue

        high_rating = high_rating.sort_values("timestamp")
        n_high = len(high_rating)
        n_valid = max(1, n_high // 2)

        valid_part = high_rating.iloc[:n_valid]
        test_part = high_rating.iloc[n_valid:]

        train_rows.append(train_part)
        if len(low_rating) > 0:
            train_rows.append(low_rating)
        valid_rows.append(valid_part)
        test_rows.append(test_part)

    train_df = pd.concat(train_rows, axis=0).reset_index(drop=True)
    valid_df = pd.concat(valid_rows, axis=0).reset_index(drop=True) if valid_rows else pd.DataFrame()
    test_df = pd.concat(test_rows, axis=0).reset_index(drop=True) if test_rows else pd.DataFrame()

    return train_df, valid_df, test_df


def build_user_history(train_df):
    user_history = {}
    for user_id, group in train_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        user_history[int(user_id)] = group["item_id"].astype(int).tolist()
    return user_history


def build_user_seen_items(df, n_users):
    seen_items = [set() for _ in range(n_users + 1)]
    for row in df.itertuples(index=False):
        seen_items[int(row.user_id)].add(int(row.item_id))
    return seen_items


def build_train_uir(train_df):
    return train_df[["user_id", "item_id", "rating", "timestamp"]].to_numpy(dtype=np.float64)


def build_ui(df):
    return df[["user_id", "item_id", "timestamp"]].to_numpy(dtype=np.float64)