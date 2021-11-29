import pandas as pd
from argparse import ArgumentParser


train = "train.csv"
test = "test.csv"
test_labels = "test_labels.csv"
test_clean = "test_clean.csv"
text_col = "comment_text"

def split_train():
    train_df = pd.read_csv(train)
    toxic = train_df[train_df["toxic"] == 1]
    toxic.to_csv("train_all_toxic.csv", index=False)

    nontoxic = train_df[train_df["toxic"] == 0]
    nontoxic.to_csv("train_all_nontoxic.csv", index=False)

# split_train()

def clean_test():
    test_df = pd.read_csv(test)
    test_labels_df = pd.read_csv(test_labels)
    test_df["toxic"] = test_labels_df["toxic"]
    cleaned = test_df[test_df["toxic"] != -1]
    cleaned.to_csv(test_clean, index=False)

    toxic = cleaned[cleaned["toxic"] == 1]
    toxic.to_csv("test_all_toxic.csv", index=False)

    nontoxic = cleaned[cleaned["toxic"] == 0]
    nontoxic.to_csv("test_all_nontoxic.csv", index=False)

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

def process_short(size=100, ratio=None, mix=True, type='train'):
    if ratio is not None:
        size = ratio * len(pd.read_csv(f'{type}_all_nontoxic.csv'))
        size = int(size)
    if mix:
        df = pd.concat([pd.read_csv(f"{type}_all_nontoxic.csv").head(size//2), pd.read_csv(f"{type}_all_toxic.csv").head(size//2)])
        new_df = augment(df)[['prefix', 'toxic']]
        new_df.to_csv(f'augmented_{type}_mixed.csv', index=False)
    else:
        df = pd.read_csv(f'{type}_all_nontoxic.csv').head(size)
        new_df = augment(df)[['prefix', 'toxic']]
        new_df.to_csv(f'augmented_{type}.csv', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    # DATA
    parser.add_argument('--size', type=int, default=100) # ideally 5000 sentences for 100K prefixes
    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--mix', type=bool, default=True, help='whether or not to evenly mix train and test')
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test', 'both'])

    args = parser.parse_args()
    if args.type == 'both':
        process_short(size=args.size, ratio=args.ratio, mix=args.mix, type='train')
        process_short(size=args.size, ratio=args.ratio, mix=args.mix, type='test') 
    else:
        process_short(size=args.size, ratio=args.ratio, mix=args.mix, type=args.type)
