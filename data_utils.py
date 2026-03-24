import pandas as pd
import numpy as np


def load_ratings(path):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="::", engine="python", names=cols)
    df = df.sort_values(["user_id", "timestamp"]).copy()
    return df


def leave_one_out_split(df):
    """
    For each user:
    - all but last 2 interactions -> train
    - second last -> valid
    - last -> test
    Users with fewer than 3 interactions are skipped.
    """
    train_rows = []
    valid_rows = []
    test_rows = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            continue

        train_rows.append(group.iloc[:-2])
        valid_rows.append(group.iloc[[-2]])
        test_rows.append(group.iloc[[-1]])

    train_df = pd.concat(train_rows, axis=0).reset_index(drop=True)
    valid_df = pd.concat(valid_rows, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_rows, axis=0).reset_index(drop=True)

    return train_df, valid_df, test_df


def build_user_history(train_df):
    """
    user_history[u] = ordered list of interacted item ids in training data
    """
    user_history = {}
    for user_id, group in train_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        user_history[int(user_id)] = group["item_id"].astype(int).tolist()
    return user_history


def build_user_seen_items(df, n_users):
    """
    seen_items[u] = set of items user u has interacted with in df
    user ids are assumed to start from 1
    """
    seen_items = [set() for _ in range(n_users + 1)]
    for row in df.itertuples(index=False):
        seen_items[int(row.user_id)].add(int(row.item_id))
    return seen_items


def build_train_uir(train_df):
    return train_df[["user_id", "item_id", "rating"]].to_numpy(dtype=np.float32)


def build_ui(df):
    return df[["user_id", "item_id"]].to_numpy(dtype=np.int64)


def get_num_users_items(df):
    n_users = int(df["user_id"].max())
    n_items = int(df["item_id"].max())
    return n_users, n_items
