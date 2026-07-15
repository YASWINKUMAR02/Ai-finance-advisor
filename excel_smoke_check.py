import pandas as pd

from data_loader import load_transactions


def main() -> None:
    dataframe = pd.read_csv("data/sample_transactions.csv")
    dataframe.to_excel("data/sample_transactions.xlsx", index=False)
    loaded = load_transactions("data/sample_transactions.xlsx")
    print(len(loaded))
    print(list(loaded.columns))


if __name__ == "__main__":
    main()
