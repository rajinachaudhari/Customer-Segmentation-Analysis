
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

pd.set_option('display.max_columns', 500)

# 👇 Run independently
if __name__ == "__main__":
    df = load_data("dataset/fintech_wallet_users_sample.csv")
    print(df.head())
    print(df.shape)
