import pandas as pd
import os

# define file path
input_files = {
    "SPY": "../Dataset/SPY_alpha.csv",
    "GLD": "../Dataset/GLD_alpha.csv",
    "BND": "../Dataset/BND_alpha.csv",
    "VIOO": "../Dataset/VIOO_alpha.csv"
}
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# process dataset, the start time based on VIOO
def preprocess_etf(file_path, etf_name, start_date="2010-09-09"):
    # read data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # filter data
    df = df[df['Date'] >= start_date]
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    # sort data and drop the duplicated data
    df = df.drop_duplicates().sort_values("Date")

    # fill the missing data
    df = df.ffill().bfill()

    # reset the index
    df.reset_index(drop=True, inplace=True)

    # save the processed_data as a new CSV file
    output_path = os.path.join(output_dir, f"{etf_name}_processed.csv")
    df.to_csv(output_path, index=False)

    print(f"{etf_name} processing finished, saved to {output_path}")
    return df


processed_data = {}
for name, file in input_files.items():
    processed_data[name] = preprocess_etf(file, name)