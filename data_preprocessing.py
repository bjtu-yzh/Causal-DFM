import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


class StableDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.imputer = None
        self.feature_columns = None

    def zscore_cross_section(self, data):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        result = data.copy()
        T = data.shape[0]

        for t in range(T):
            if data.ndim == 2:
                col = data[t]
                mean = np.mean(col)
                std = np.std(col)
                if std > 1e-8:
                    result[t] = (col - mean) / std
                else:
                    result[t] = 0.0
            else:
                for p in range(data.shape[2]):
                    col = data[t, :, p]
                    mean = np.mean(col)
                    std = np.std(col)
                    if std > 1e-8:
                        result[t, :, p] = (col - mean) / std
                    else:
                        result[t, :, p] = 0.0

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        result = np.clip(result, -10, 10)

        return result

    def load_npy_data(self):
        print("Loading npy data...")
        try:
            numpy_version = np.__version__
            print(f"    NumPy version: {numpy_version}")

            try:
                features_raw = np.load(self.config.price_and_factor_path, allow_pickle=True)
            except (ModuleNotFoundError, ImportError) as e:
                if 'numpy._core' in str(e) or '_core' in str(e):
                    print(f"    NumPy version compatibility issue: {e}")
                    print(f"    Suggested fix: pip install --upgrade numpy")
                    print(f"    Or: conda install numpy")
                    print(f"    Trying alternative loading method...")
                    import importlib
                    importlib.reload(np)
                    features_raw = np.load(self.config.price_and_factor_path, allow_pickle=True)
                else:
                    raise

            try:
                returns_raw = np.load(self.config.return_path, allow_pickle=True)
            except (ModuleNotFoundError, ImportError) as e:
                if 'numpy._core' in str(e) or '_core' in str(e):
                    import importlib
                    importlib.reload(np)
                    returns_raw = np.load(self.config.return_path, allow_pickle=True)
                else:
                    raise

            try:
                timestamps_raw = np.load(self.config.time_stamp_path, allow_pickle=True)
            except (ModuleNotFoundError, ImportError) as e:
                if 'numpy._core' in str(e) or '_core' in str(e):
                    import importlib
                    importlib.reload(np)
                    timestamps_raw = np.load(self.config.time_stamp_path, allow_pickle=True)
                else:
                    raise

            print(f"Raw data shapes:")
            print(f"  features: {features_raw.shape}")
            print(f"  returns: {returns_raw.shape}")
            print(f"  timestamps: {timestamps_raw.shape}")

            N, T, P = features_raw.shape
            assert returns_raw.shape == (N,
                                         T), f"Returns shape {returns_raw.shape} doesn't match features shape {features_raw.shape}"

            if timestamps_raw.ndim == 1:
                timestamps = timestamps_raw
            elif timestamps_raw.ndim == 2:
                if timestamps_raw.shape == (N, T):
                    timestamps = timestamps_raw[0]
                elif timestamps_raw.shape == (T, N):
                    timestamps = timestamps_raw[:, 0]
                else:
                    print(f"Warning: timestamps shape {timestamps_raw.shape} unexpected, using first row")
                    timestamps = timestamps_raw[0] if timestamps_raw.shape[0] == T else timestamps_raw[:, 0]
            else:
                print(f"Warning: timestamps has {timestamps_raw.ndim} dimensions, using first element")
                timestamps = timestamps_raw.flatten()[:T]

            features = features_raw.transpose(1, 0, 2)
            returns = returns_raw.transpose(1, 0)

            print(f"Transposed data shapes:")
            print(f"  features: {features.shape}")
            print(f"  returns: {returns.shape}")
            print(f"  timestamps: {timestamps.shape}")

            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

            features = np.clip(features, -10, 10)
            returns = np.clip(returns, -10, 10)

            return features, returns, timestamps

        except Exception as e:
            print(f"Error loading npy data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def create_sequences(self, features, returns, timestamps):
        print("Creating sequences from npy data...")
        T, N, P = features.shape

        all_sequences = []
        all_targets = []
        all_hist_returns = []
        all_dates = []
        all_time_indices = []
        all_stock_indices = []

        filtered_target = 0
        filtered_hist_return = 0
        filtered_sequence = 0

        seq_length = self.config.seq_length

        for t in tqdm(range(seq_length, T), desc="Processing time steps"):
            hist_features = features[t - seq_length:t]

            hist_returns_window = returns[t - seq_length:t]

            current_returns = returns[t]

            for n in range(N):
                stock_seq = hist_features[:, n, :]

                stock_hist_returns = hist_returns_window[:, n]

                target_return = current_returns[n]

                hist_return = stock_hist_returns[-1]

                if t < len(timestamps):
                    date_val = timestamps[t]
                else:
                    date_val = timestamps[-1] if len(timestamps) > 0 else None

                if np.isnan(target_return) or np.isinf(target_return):
                    target_return = 0.0
                    filtered_target += 1

                if np.isnan(hist_return) or np.isinf(hist_return):
                    hist_return = 0.0
                    filtered_hist_return += 1

                if np.isnan(stock_seq).any() or np.isinf(stock_seq).any():
                    stock_seq = np.nan_to_num(stock_seq, nan=0.0, posinf=0.0, neginf=0.0)
                    filtered_sequence += 1

                all_sequences.append(stock_seq)
                all_targets.append(target_return)
                all_hist_returns.append(hist_return)

                all_time_indices.append(t)
                all_stock_indices.append(n)
                all_dates.append((t, n, date_val))

        if len(all_sequences) == 0:
            print("Warning: No valid sequences created!")
            return np.array([]), np.array([]), np.array([]), []

        sequences_array = np.array(all_sequences)
        targets_array = np.array(all_targets)
        hist_returns_array = np.array(all_hist_returns)

        print(f"Created {len(all_sequences)} sequences")
        print(f"  Sequence shape: {sequences_array.shape}")
        print(f"  Target shape: {targets_array.shape}")
        print(f"  Hist returns shape: {hist_returns_array.shape}")
        print(f"\nOutlier handling statistics (fixed, not filtered):")
        print(
            f"  Target return outliers fixed: {filtered_target} samples (NaN/Inf filled with 0, preserving real volatility range)")
        print(
            f"  Historical return outliers fixed: {filtered_hist_return} samples (NaN/Inf filled with 0, preserving real volatility range)")
        print(f"  Sequence outliers fixed: {filtered_sequence} samples (NaN/Inf filled with 0)")
        print(f"  Total fixed samples: {filtered_target + filtered_hist_return + filtered_sequence} samples")
        print(f"  Retained samples: {len(all_sequences)} samples (all samples retained, return range not clipped)")

        return sequences_array, targets_array, hist_returns_array, all_dates, all_time_indices, all_stock_indices

    def train_test_split_temporal(self, sequences, targets, hist_returns, dates, time_indices, stock_indices):
        n_samples = len(sequences)
        if n_samples == 0:
            empty = np.array([])
            return (empty, empty, empty, empty, empty), (empty, empty, empty, empty, empty), (empty, empty, empty,
                                                                                              empty, empty)

        n_train = int(n_samples * self.config.train_ratio)
        n_val = int(n_samples * self.config.val_ratio)

        train_seq = sequences[:n_train]
        train_target = targets[:n_train]
        train_hist_returns = hist_returns[:n_train]
        train_time_indices = time_indices[:n_train] if time_indices else None
        train_stock_indices = stock_indices[:n_train] if stock_indices else None

        val_seq = sequences[n_train:n_train + n_val]
        val_target = targets[n_train:n_train + n_val]
        val_hist_returns = hist_returns[n_train:n_train + n_val]
        val_time_indices = time_indices[n_train:n_train + n_val] if time_indices else None
        val_stock_indices = stock_indices[n_train:n_train + n_val] if stock_indices else None

        test_seq = sequences[n_train + n_val:]
        test_target = targets[n_train + n_val:]
        test_hist_returns = hist_returns[n_train + n_val:]
        test_time_indices = time_indices[n_train + n_val:] if time_indices else None
        test_stock_indices = stock_indices[n_train + n_val:] if stock_indices else None

        return (train_seq, train_target, train_hist_returns, train_time_indices, train_stock_indices), \
            (val_seq, val_target, val_hist_returns, val_time_indices, val_stock_indices), \
            (test_seq, test_target, test_hist_returns, test_time_indices, test_stock_indices)

    def fit_imputer(self, train_data):
        if len(train_data) == 0:
            print("Warning: No training data for imputer")
            return

        print("Fitting imputer...")
        train_2d = train_data.reshape(-1, train_data.shape[-1])

        if self.config.impute_method == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif self.config.impute_method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)

        self.imputer.fit(train_2d)

    def fit_scaler(self, train_data):
        if len(train_data) == 0:
            print("Warning: No training data for scaler")
            return

        print("Fitting scaler...")
        train_2d = train_data.reshape(-1, train_data.shape[-1])

        if self.config.normalize_method == 'standard':
            self.scaler = StandardScaler()
        elif self.config.normalize_method == 'minmax':
            self.scaler = StandardScaler()

        self.scaler.fit(train_2d)

    def transform_data(self, data):
        if len(data) == 0:
            return data

        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])

        if self.imputer is not None:
            data_2d = self.imputer.transform(data_2d)

        if self.scaler is not None:
            data_2d = self.scaler.transform(data_2d)

        return data_2d.reshape(original_shape)

    def preprocess_all_data(self):
        try:
            features, returns, timestamps = self.load_npy_data()
            if features is None or returns is None:
                raise ValueError("Failed to load npy data")

            T, N, P = features.shape
            self.feature_columns = [f"feature_{i}" for i in range(P)]
            print(f"Number of features: {P}")
            print(f"Number of stocks: {N}")
            print(f"Number of time steps: {T}")

            print("\nApplying cross-sectional normalization...")
            features = self.zscore_cross_section(features)
            returns = self.zscore_cross_section(returns)

            sequences, targets, hist_returns, dates, time_indices, stock_indices = self.create_sequences(features,
                                                                                                         returns,
                                                                                                         timestamps)
            if len(sequences) == 0:
                raise ValueError("No valid sequences created")

            print(f"\nTotal sequences: {sequences.shape}")

            (train_seq, train_target, train_hist_returns, train_time_indices, train_stock_indices), \
                (val_seq, val_target, val_hist_returns, val_time_indices, val_stock_indices), \
                (test_seq, test_target, test_hist_returns, test_time_indices, test_stock_indices) = \
                self.train_test_split_temporal(sequences, targets, hist_returns, dates, time_indices, stock_indices)

            print(f"Train sequences: {train_seq.shape}")
            print(f"Val sequences: {val_seq.shape}")
            print(f"Test sequences: {test_seq.shape}")

            train_seq_processed = train_seq if len(train_seq) > 0 else train_seq
            val_seq_processed = val_seq if len(val_seq) > 0 else val_seq
            test_seq_processed = test_seq if len(test_seq) > 0 else test_seq

            return {
                'train': (train_seq_processed, train_target, train_hist_returns, train_time_indices,
                          train_stock_indices),
                'val': (val_seq_processed, val_target, val_hist_returns, val_time_indices, val_stock_indices),
                'test': (test_seq_processed, test_target, test_hist_returns, test_time_indices, test_stock_indices),
                'feature_columns': self.feature_columns,
                'dates': dates
            }

        except Exception as e:
            print(f"Error in preprocess_all_data: {e}")
            import traceback
            traceback.print_exc()
            empty_array = np.array([])
            return {
                'train': (empty_array, empty_array, empty_array, [], []),
                'val': (empty_array, empty_array, empty_array, [], []),
                'test': (empty_array, empty_array, empty_array, [], []),
                'feature_columns': [],
                'dates': []
            }


class StableStockDataset(Dataset):
    def __init__(self, sequences, targets, hist_returns, time_indices=None, stock_indices=None,
                 use_temporal_pairs=False):
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        hist_returns = np.nan_to_num(hist_returns, nan=0.0, posinf=0.0, neginf=0.0)
        sequences = np.clip(sequences, -10, 10)
        targets = np.clip(targets, -10, 10)
        hist_returns = np.clip(hist_returns, -10, 10)
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.hist_returns = torch.FloatTensor(hist_returns)

        self.time_indices = np.array(time_indices) if time_indices is not None else None
        self.stock_indices = np.array(stock_indices) if stock_indices is not None else None

        self.use_temporal_pairs = use_temporal_pairs
        if self.use_temporal_pairs and self.time_indices is not None:
            self.time_to_indices = {}
            for i, t_idx in enumerate(self.time_indices):
                if t_idx not in self.time_to_indices:
                    self.time_to_indices[t_idx] = []
                self.time_to_indices[t_idx].append(i)
            self.sorted_time_indices = sorted(self.time_to_indices.keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if not self.use_temporal_pairs:
            return self.sequences[idx], self.targets[idx], self.hist_returns[idx]
        else:
            if self.time_indices is None:
                return (
                    self.sequences[idx],
                    self.targets[idx],
                    self.hist_returns[idx],
                    torch.zeros_like(self.sequences[idx]),
                    torch.zeros_like(self.targets[idx]),
                    torch.zeros_like(self.hist_returns[idx])
                )

            t_idx = self.time_indices[idx]
            t_prev_idx = t_idx - 1

            prev_idx = None
            if t_prev_idx in self.time_to_indices:
                if self.stock_indices is not None:
                    stock_idx = self.stock_indices[idx]
                    for candidate_idx in self.time_to_indices[t_prev_idx]:
                        if self.stock_indices[candidate_idx] == stock_idx:
                            prev_idx = candidate_idx
                            break

                if prev_idx is None:
                    prev_idx = self.time_to_indices[t_prev_idx][0]

            if prev_idx is not None:
                return (
                    self.sequences[idx],
                    self.targets[idx],
                    self.hist_returns[idx],
                    self.sequences[prev_idx],
                    self.targets[prev_idx],
                    self.hist_returns[prev_idx]
                )
            else:
                return (
                    self.sequences[idx],
                    self.targets[idx],
                    self.hist_returns[idx],
                    torch.zeros_like(self.sequences[idx]),
                    torch.zeros_like(self.targets[idx]),
                    torch.zeros_like(self.hist_returns[idx])
                )