import pandas as pd

train = "train.csv"
test = "test.csv"
test_labels = "test_labels.csv"
test_clean = "test_clean.csv"
text_col = "comment_text"

def clean_test():
    test_df = pd.read_csv(test)
    test_labels_df = pd.read_csv(test_labels)
    test_df["toxic"] = test_labels_df["toxic"]
    cleaned = test_df[test_df["toxic"] != -1]
    cleaned.to_csv(test_clean, index=False)

    toxic = cleaned[cleaned["toxic"] == 1]
    toxic.to_csv("test_all_toxic.csv", index=False)

# clean_test()

def row_to_prefix_row(row):
    prefixes = []
    words = row[text_col].split()
    prefix = ""
    for w in words:
        prefix += w
        prefixes.append(prefix)
        prefix += " "
    return pd.Series(prefixes)

def augment(df):
    new_df = df.apply(row_to_prefix_row, axis=1)
    print("Split complete")
    new_df["id"] = df["id"]
    new_df["toxic"] = df["toxic"]
    cols = ["id", "toxic"]
    new_df = new_df.set_index(cols).stack().reset_index()
    print("Stack complete")
    del new_df["level_2"]
    new_df.columns = ["id", "toxic", "prefix"]
    return new_df

def process_full():
    train_df = pd.read_csv(train)
    new_train = augment(train_df)
    new_train.to_csv("augmented_train.csv", index=False)

    # test_df = pd.read_csv(test_clean)
    # new_test = augment(test_df)
    # new_test.to_csv("augmented_test.csv", index=False)

# process_full()

def process_short(size, mix=True):
    train_df = pd.read_csv(train).head(size)
    new_train = augment(train_df)
    new_train.to_csv("augmented_train_short.csv", index=False)

    if mix:
        test_df = pd.concat([pd.read_csv(test_clean).head(size//2), pd.read_csv("test_all_toxic.csv").head(size//2)])
        new_test = augment(test_df)
        new_test.to_csv("augmented_test_short_mixed.csv", index=False)
    else:
        test_df = pd.read_csv(test_clean).head(size)
        new_test = augment(test_df)
        new_test.to_csv("augmented_test_short.csv", index=False)

# process_short(100)

