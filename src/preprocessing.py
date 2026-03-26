import pandas as pd

def clean_data(df):
    df = df.copy()

    pd.set_option('display.max_columns', 500)

    #Descriptive Analysis
    print("df type:", type(df))     #type() function returns the type of the object passed to it
    print("df shape:", df.shape) 
    print(df.head()) #prints first 5 rows
    print(df.info()) #prints summary info about the dataframe like notnull,datatypes,memory usage
    print(df.describe()) #prints statistical summary() of numerical columns
 

    # Drop useless columns
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Convert date
    df["registration_date"] = pd.to_datetime(df["registration_date"])

    return df


def handle_outliers(df):
    df = df.copy()

    # Clip extreme values (IQR-based)
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df
if __name__ == "__main__":
    raw_df = pd.read_csv("dataset/fintech_wallet_users_sample.csv")
    clean_df = clean_data(raw_df)
    final_df = handle_outliers(clean_df)
    print(final_df.head())
    print(final_df.shape)
    final_df.to_csv("fintech_wallet_users_cleaned.csv", index=False)