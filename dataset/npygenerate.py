import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("Ignore")

files = os.listdir(".")
csv_files = [file for file in files if file.endswith(".csv")]

print(f"Found CSV files: {csv_files}")

stock_codes = []
for file in csv_files:
    code_without_ext = file.replace('.csv', '')
    stock_codes.append(code_without_ext)

print(f"Extracted stock codes: {stock_codes}")

drop_stock_list = [
    '001391.XSHE',
    '601059.XSHG',
    '601136.XSHG',
    '603296.XSHG',
    '688472.XSHG',
    '688506.XSHG',
    "302132.XSHE",
]

stock_codes = [code for code in stock_codes if code not in drop_stock_list]

existing_csv_files = []
valid_stock_codes = []
for code in stock_codes:
    possible_files = [
        f"{code}.csv",
        f"{code}.csv",
    ]

    found = False
    for csv_file in possible_files:
        if csv_file in csv_files:
            existing_csv_files.append(csv_file)
            valid_stock_codes.append(code)
            found = True
            break

    if not found:
        print(f"Warning: CSV file for stock code {code} does not exist, skipped")

if len(existing_csv_files) == 0:
    raise FileNotFoundError("No processable CSV files found, please check file paths and names")

print(f"CSV files to be processed: {existing_csv_files}")
print(f"Corresponding stock codes: {valid_stock_codes}")

data_list = []
for csv_file in existing_csv_files:
    try:
        df = pd.read_csv(csv_file)
        data_list.append(df)
        print(f"Successfully read file: {csv_file}, shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file {csv_file}: {e}")

stock_codes = valid_stock_codes

shapes = [df.shape for df in data_list]
print(f"Shapes of stock data: {shapes}")

min_rows = min(shape[0] for shape in shapes)
min_cols = min(shape[1] for shape in shapes)
print(f"Minimum rows: {min_rows}, minimum columns: {min_cols}")

dates_list = data_list[0]['tradeDate'].tolist()[:min_rows] if len(data_list) > 0 else []

if len(data_list) > 0:
    all_columns = data_list[0].columns.tolist()
    exclude_columns = {all_columns[0]}
    if 'close' in all_columns:
        exclude_columns.add('close')
    else:
        raise ValueError("Column named 'close' not found, please check data column names")
    factor_columns = [col for col in all_columns if col not in exclude_columns]
    factor_names = factor_columns
else:
    factor_names = []
    factor_columns = []

print(f"Number of stocks: {len(stock_codes)}")
print(f"Number of trading days: {len(dates_list)}")
print(f"Number of factors: {len(factor_names)}")

stocks_data = []

for i, stock_code in enumerate(stock_codes):
    stock_df = data_list[i]
    stock_df = stock_df.iloc[:min_rows, :min_cols]
    stock_df = stock_df.ffill()
    missing_count = stock_df.isna().sum().sum()
    print(f"Number of missing values for stock {stock_code}: {missing_count}")
    numerical_data = stock_df.iloc[:, 1:].values.astype(np.float64)
    time_indices = np.arange(len(numerical_data)).reshape(-1, 1)
    stock_array = np.concatenate([time_indices, numerical_data], axis=1)
    stocks_data.append(stock_array)

stocks_array = np.array(stocks_data)

print(f"stocks_array shape: {stocks_array.shape}")

np.save("stocks_cleaned.npy", stocks_array)

df = np.load("stocks_cleaned.npy", allow_pickle=True)

if len(data_list) > 0:
    all_columns = data_list[0].columns.tolist()
    close_col_name = 'close'
    if close_col_name in all_columns:
        close_col_idx = all_columns.index(close_col_name)
        close_array_idx = close_col_idx
    else:
        raise ValueError("Column named 'close' not found, cannot extract closing prices")
else:
    raise ValueError("data_list is empty, cannot extract closing prices")

close_price = df[:, :, close_array_idx]

r = (close_price[:, 1:] / close_price[:, :-1]) - 1

time_stamp = df[:, :, 0]

factor_and_price = df[:, :, 1:].astype(np.float64, copy=True)

r = r[:, 1:].astype(np.float64, copy=True)

print(f"Return shape: {r.shape}")
print(f"Timestamp shape: {time_stamp.shape}")
print(f"Factor and price shape: {factor_and_price.shape}")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

train_size = round(TRAIN_RATIO * factor_and_price.shape[1])

train_factor_and_price = factor_and_price[:, :train_size, :]
N, T, F = train_factor_and_price.shape
train_max = np.nanmax(train_factor_and_price.reshape(N * T, F), axis=0)
train_min = np.nanmin(train_factor_and_price.reshape(N * T, F), axis=0)

normalized_data = (factor_and_price - train_min) / (train_max - train_min)
fill_data = np.nan_to_num(normalized_data, copy=True, nan=0)

np.save("price_and_factor.npy", fill_data)
np.save("return.npy", r)
np.save("time_stamp.npy", time_stamp)

print("Data processing completed!")
print(f"Saved files:")
print(f"- stocks_cleaned.npy: shape {df.shape}")
print(f"- price_and_factor.npy: shape {fill_data.shape}")
print(f"- return.npy: shape {r.shape}")
print(f"- time_stamp.npy: shape {time_stamp.shape}")

if len(stock_codes) > 0:
    print(f"\nData for first stock:")
    print(f"   Timestamp range: {time_stamp[0, 0]:.0f} to {time_stamp[0, -1]:.0f}")
    print(f"   Closing price range: {close_price[0].min():.2f} to {close_price[0].max():.2f}")
    print(f"   Return range: {r[0].min():.4f} to {r[0].max():.4f}")

import numpy as np

stocks_cleaned = np.load("stocks_cleaned.npy")
print(f"stocks_cleaned shape: {stocks_cleaned.shape}")

close_price = stocks_cleaned[:, :, -1]
print(f"Closing price shape: {close_price.shape}")

time_indices = stocks_cleaned[:, :, 0]
print(f"Timestamp shape: {time_indices.shape}")

r_temp = (close_price[:, 1:] / close_price[:, :-1]) - 1
print(f"Original return shape: {r_temp.shape}")

r_padded = np.zeros_like(close_price)
r_padded[:, 1:] = r_temp
r_padded[:, 0] = np.nan
print(f"Padded return shape: {r_padded.shape}")

r_padded2 = np.full_like(close_price, np.nan)
r_padded2[:, :-1] = r_temp
print(f"Alternative padded return shape: {r_padded2.shape}")

time_indices_trimmed = time_indices[:, 1:]
print(f"Trimmed timestamp shape: {time_indices_trimmed.shape}")
print(f"Current return shape: {r_temp.shape}")
print(f"Shapes consistent: {time_indices_trimmed.shape == r_temp.shape}")

print("\nSaving aligned data...")

np.save("time_stamp.npy", time_indices_trimmed)
np.save("return.npy", r_temp)

factor_and_price = stocks_cleaned[:, 1:, 1:]
print(f"Trimmed factor and price shape: {factor_and_price.shape}")
print(f"Consistent with return shape: {factor_and_price.shape[:2] == r_temp.shape}")

np.save("price_and_factor.npy", factor_and_price)

print("\nData shape summary:")
print(f"Timestamp: {time_indices_trimmed.shape}")
print(f"Return: {r_temp.shape}")
print(f"Factor and price: {factor_and_price.shape}")
print("All data now aligned in time dimension!")