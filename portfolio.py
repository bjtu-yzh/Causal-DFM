import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from config import StableConfig
from data_preprocessing import StableDataPreprocessor, StableStockDataset
from models import StableCaRIVAEWithDiffusion
from evaluation import test_model

class EnhancedPortfolioManager:
    def __init__(self, portfolio_size=50, max_turnover=10, transaction_cost=0.003, random_seed=None):
        self.portfolio_size = portfolio_size
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.current_portfolio = None
        self.portfolio_value = 1.0
        self.cash = 0.0
        self.portfolio_history = []
        self.daily_returns = []
        self.turnover_history = []
        self.transaction_cost_history = []

    def initialize_portfolio_random(self, date_index, predictions, available_stocks=None):
        if available_stocks is None:
            valid_indices = np.where(np.isfinite(predictions))[0]
            available_stocks = valid_indices.tolist()

        if len(available_stocks) < self.portfolio_size:
            selected_stocks = available_stocks.copy()
        else:
            selected_stocks = random.sample(available_stocks, self.portfolio_size)

        self.current_portfolio = selected_stocks
        self._record_portfolio_state(date_index, "RANDOM_INIT")
        return selected_stocks

    def rebalance_portfolio(self, date_index, predictions, actual_returns):
        if self.current_portfolio is None:
            return self.initialize_portfolio_random(date_index, predictions)

        daily_return = self._calculate_daily_return(actual_returns)
        self.portfolio_value *= (1 + daily_return)
        self.daily_returns.append(daily_return)

        valid_indices = np.where(np.isfinite(predictions))[0]
        if len(valid_indices) < self.portfolio_size:
            new_stocks = valid_indices.tolist()
        else:
            sorted_indices = np.argsort(predictions[valid_indices])[::-1]
            new_stocks = valid_indices[sorted_indices[:self.portfolio_size]].tolist()

        current_set = set(self.current_portfolio)
        new_set = set(new_stocks)

        stocks_to_sell = current_set - new_set
        stocks_to_buy = new_set - current_set

        max_changes = min(self.max_turnover, len(stocks_to_sell), len(stocks_to_buy))

        stocks_to_sell_final = []
        stocks_to_buy_final = []

        if max_changes > 0:
            sell_candidates = list(stocks_to_sell)
            sell_predictions = predictions[list(sell_candidates)]
            sell_priority = np.argsort(sell_predictions)
            stocks_to_sell_final = [sell_candidates[i] for i in sell_priority[:max_changes]]

            buy_candidates = list(stocks_to_buy)
            buy_predictions = predictions[list(buy_candidates)]
            buy_priority = np.argsort(buy_predictions)[::-1]
            stocks_to_buy_final = [buy_candidates[i] for i in buy_priority[:max_changes]]

            final_portfolio = current_set - set(stocks_to_sell_final)
            final_portfolio = final_portfolio.union(set(stocks_to_buy_final))

            while len(final_portfolio) < self.portfolio_size and len(current_set) > 0:
                retain_candidates = list(current_set - final_portfolio)
                if len(retain_candidates) > 0:
                    retain_predictions = predictions[retain_candidates]
                    retain_priority = np.argsort(retain_predictions)[::-1]
                    additional_stocks = [retain_candidates[i] for i in retain_priority[:min(
                        self.portfolio_size - len(final_portfolio), len(retain_candidates))]]
                    final_portfolio = final_portfolio.union(set(additional_stocks))
                else:
                    break
        else:
            final_portfolio = current_set

        transaction_cost = self._calculate_transaction_cost(current_set, final_portfolio)
        self.portfolio_value -= transaction_cost
        self.transaction_cost_history.append(transaction_cost)

        self.turnover_history.append(len(stocks_to_sell_final) + len(stocks_to_buy_final))

        self.current_portfolio = list(final_portfolio)
        self._record_portfolio_state(date_index, "REBALANCE")

        return self.current_portfolio

    def _calculate_daily_return(self, actual_returns):
        if self.current_portfolio is None:
            return 0.0

        weight = 1.0 / len(self.current_portfolio)
        portfolio_return = 0.0

        for stock_idx in self.current_portfolio:
            if stock_idx < len(actual_returns) and np.isfinite(actual_returns[stock_idx]):
                log_return = actual_returns[stock_idx]
                simple_return = np.exp(log_return) - 1
                portfolio_return += weight * simple_return

        return portfolio_return

    def _calculate_transaction_cost(self, old_portfolio, new_portfolio):
        turnover_stocks = len((old_portfolio - new_portfolio).union(new_portfolio - old_portfolio))
        turnover_value = (turnover_stocks / self.portfolio_size) * self.portfolio_value
        transaction_cost = turnover_value * self.transaction_cost
        return transaction_cost

    def _record_portfolio_state(self, date_index, action):
        self.portfolio_history.append({
            'date_index': date_index,
            'portfolio_value': self.portfolio_value,
            'num_stocks': len(self.current_portfolio) if self.current_portfolio else 0,
            'action': action,
            'cash': self.cash
        })

    def get_portfolio_history_df(self):
        return pd.DataFrame(self.portfolio_history)

    def get_performance_metrics(self):
        if len(self.daily_returns) == 0:
            return {}

        daily_returns_array = np.array(self.daily_returns)

        total_return = self.portfolio_value - 1.0

        if len(daily_returns_array) > 0:
            annual_return = (1 + total_return) ** (252 / len(daily_returns_array)) - 1
        else:
            annual_return = 0.0

        if len(daily_returns_array) > 1:
            annual_volatility = np.std(daily_returns_array) * np.sqrt(252)
        else:
            annual_volatility = 0.0

        if annual_volatility > 0:
            sharpe_ratio = annual_return / annual_volatility
        else:
            sharpe_ratio = 0.0

        portfolio_values = [1.0]
        current_value = 1.0
        for ret in self.daily_returns:
            current_value *= (1 + ret)
            portfolio_values.append(current_value)

        portfolio_values_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        avg_turnover = np.mean(self.turnover_history) if self.turnover_history else 0.0
        total_transaction_cost = np.sum(self.transaction_cost_history)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover,
            'total_transaction_cost': total_transaction_cost,
            'final_portfolio_value': self.portfolio_value
        }

    def reset(self):
        self.current_portfolio = None
        self.portfolio_value = 1.0
        self.cash = 0.0
        self.portfolio_history = []
        self.daily_returns = []
        self.turnover_history = []
        self.transaction_cost_history = []

class MultiPortfolioTester:
    def __init__(self, num_tests=50):
        self.num_tests = num_tests
        self.best_portfolio = None
        self.best_final_value = 0.0
        self.all_results = []

    def run_multiple_tests(self, model, test_loader, config, test_indices):
        print(f"Running {self.num_tests} random portfolio tests...")

        device = config.device
        model.eval()

        all_predictions = []
        all_targets = []
        all_time_indices = []

        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Computing predictions"):
                try:
                    if len(batch_data) == 6:
                        x_batch, y_batch, r_hist_batch, x_t_prev, r_t_prev, r_hist_t_prev = batch_data
                    else:
                        x_batch, y_batch, r_hist_batch = batch_data

                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    if hasattr(model, 'diffusion'):
                        model.reset_f_prev_buffer()
                        predictions, _ = model.predict(x_batch, r_t=r_hist_batch.to(
                            device) if r_hist_batch is not None else None)
                        model.reset_f_prev_buffer()
                    else:
                        x_flat = x_batch[:, -1, :]
                        predictions = model.predict(x_batch, x_flat, y=None,
                                                    r=r_hist_batch.to(device) if r_hist_batch is not None else None)

                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_targets.extend(y_batch.cpu().numpy().flatten())

                    dataset = test_loader.dataset
                    if hasattr(dataset, 'time_indices') and dataset.time_indices is not None:
                        batch_size = x_batch.size(0)
                        start_idx = len(all_time_indices)
                        end_idx = min(start_idx + batch_size, len(dataset.time_indices))
                        batch_time_indices = dataset.time_indices[start_idx:end_idx]
                        all_time_indices.extend(batch_time_indices)
                    else:
                        all_time_indices.extend([len(all_predictions) - 1])

                except Exception as e:
                    print(f"Prediction batch error: {e}")
                    continue

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_time_indices = np.array(all_time_indices)

        unique_times = np.unique(all_time_indices)
        T_test = len(unique_times)

        predictions_by_time = []
        targets_by_time = []

        for t in unique_times:
            mask = all_time_indices == t
            if mask.sum() > 0:
                pred_t = all_predictions[mask]
                actual_t = all_targets[mask]
                predictions_by_time.append(pred_t)
                targets_by_time.append(actual_t)

        predictions_by_time = np.array(predictions_by_time)
        targets_by_time = np.array(targets_by_time)

        print(f"Test period: {T_test} days")
        print(f"Predictions shape: {predictions_by_time.shape}")
        print(f"Targets shape: {targets_by_time.shape}")

        for test_id in range(self.num_tests):
            print(f"Running test {test_id + 1}/{self.num_tests}...")

            portfolio_manager = EnhancedPortfolioManager(
                portfolio_size=50,
                max_turnover=10,
                transaction_cost=0.003,
                random_seed=test_id
            )

            for t in range(T_test):
                if t >= len(predictions_by_time):
                    continue

                pred_t = predictions_by_time[t]
                actual_t = targets_by_time[t]

                if t == 0:
                    selected_stocks = portfolio_manager.initialize_portfolio_random(t, pred_t)
                else:
                    selected_stocks = portfolio_manager.rebalance_portfolio(t, pred_t, actual_t)

            final_value = portfolio_manager.portfolio_value
            self.all_results.append({
                'test_id': test_id,
                'final_portfolio_value': final_value,
                'total_return': final_value - 1.0,
                'num_stocks': len(portfolio_manager.current_portfolio) if portfolio_manager.current_portfolio else 0
            })

            if final_value > self.best_final_value:
                self.best_final_value = final_value
                self.best_portfolio = portfolio_manager
                print(f"New best portfolio found: final value = {final_value:.4f}")

        all_results_df = pd.DataFrame(self.all_results)
        return self.best_portfolio, all_results_df

def load_trained_model(config, model_path, input_dim):
    device = config.device

    model = StableCaRIVAEWithDiffusion(
        input_dim=input_dim,
        hidden_dims=config.vae_hidden_dims,
        z_dim=config.z_dim,
        factor_dim=config.factor_dim,
        config=config
    )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model with weights_only=False: {e}")
        checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model

def run_portfolio_experiment():
    print("=" * 70)
    print("Enhanced Portfolio Experiment with Diffusion Model")
    print("=" * 70)

    config = StableConfig()
    device = config.device
    print(f"Using device: {device}")

    model_path = "stable_cari_diffusion_transformer_complete.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None, None

    print("1. Data preprocessing...")
    preprocessor = StableDataPreprocessor(config)
    processed_data = preprocessor.preprocess_all_data()

    if len(processed_data['test'][0]) == 0:
        print("Error: No test data available")
        return None, None, None

    (test_seq, test_target, test_hist_returns, test_time_indices, test_stock_indices) = processed_data['test']

    print(f"Test data shape: {test_seq.shape}")

    print("2. Creating test dataset...")
    test_dataset = StableStockDataset(
        test_seq, test_target, test_hist_returns,
        test_time_indices, test_stock_indices,
        use_temporal_pairs=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")

    input_dim = test_seq.shape[-1] if len(test_seq) > 0 else 1

    print("3. Loading trained model...")
    model = load_trained_model(config, model_path, input_dim)

    print("4. Running multiple portfolio tests...")
    multi_tester = MultiPortfolioTester(num_tests=50)

    test_indices = test_time_indices if test_time_indices is not None else list(range(len(test_dataset)))

    best_portfolio, all_tests_df = multi_tester.run_multiple_tests(
        model, test_loader, config, test_indices
    )

    print("5. Saving results...")
    portfolio_history_df = best_portfolio.get_portfolio_history_df()
    portfolio_history_csv = "best_portfolio_daily_history.csv"
    portfolio_history_df.to_csv(portfolio_history_csv, index=False)

    performance_metrics = best_portfolio.get_performance_metrics()
    results_df = pd.DataFrame([performance_metrics])
    results_csv = "portfolio_experiment_results.csv"
    results_df.to_csv(results_csv, index=False)

    all_tests_csv = "all_portfolio_tests_results.csv"
    all_tests_df.to_csv(all_tests_csv, index=False)

    print("6. Printing results...")
    print("\n" + "=" * 80)
    print("Portfolio Experiment Final Results")
    print("=" * 80)

    print(f"Number of random tests: {50}")
    print(f"Best portfolio final value: {performance_metrics['final_portfolio_value']:.4f}")
    print(f"Best portfolio total return: {performance_metrics['total_return']:.6f}")

    for metric, value in performance_metrics.items():
        if metric in ['total_return', 'annual_return', 'annual_volatility',
                      'sharpe_ratio', 'max_drawdown', 'total_transaction_cost']:
            print(f"  {metric.replace('_', ' ').title()}: {value:.6f}")
        elif metric == 'final_portfolio_value':
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")

    print(f"\nBest portfolio daily history saved to: {portfolio_history_csv}")
    print(f"Portfolio experiment results saved to: {results_csv}")
    print(f"All portfolio tests results saved to: {all_tests_csv}")

    print("\nBest portfolio daily history preview:")
    print(portfolio_history_df.head().to_string(index=False))

    print("\nPortfolio experiment results preview:")
    print(results_df.to_string(index=False))

    print("\nAll portfolio tests results preview (top 10):")
    print(all_tests_df.head(10).to_string(index=False))

    return best_portfolio, portfolio_history_df, all_tests_df

if __name__ == "__main__":
    best_portfolio, portfolio_history, all_tests_df = run_portfolio_experiment()
    print("\nPortfolio experiment completed successfully!")