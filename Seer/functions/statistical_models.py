import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import numpy as np
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
from optuna.samplers import TPESampler
import pickle
import streamlit as st
import optuna
import plotly.graph_objs as go
from itertools import product
import concurrent.futures
from tqdm import tqdm
import statsmodels.api as sm
from joblib import Parallel, delayed
import time
import logging
from optuna.pruners import SuccessiveHalvingPruner
from tbats import TBATS
from prophet import Prophet
from bsts import BSTS

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalMethods:
    """
    Bu sınıf, verilen veri çerçevesi üzerinde zaman serisi modellerini kurmayı amaçlar.
    """

    def __init__(self, df, date_column, target_column, frequency):
        self.df = df
        self.date_column = date_column
        self.target_column = target_column
        self.frequency = frequency

    def find_data_frequency_int(self):
        """
        Verinin frekansını otomatik olarak belirler ve integer olarak çıktı verir.

        :return: Frekans değeri (integer) veya None
        """
        try:
            if self.df.index.name == self.date_column:
                self.df[self.date_column] = self.df.index

            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors='coerce')
            if self.df[self.date_column].isna().any():
                raise ValueError("Bazı tarihler datetime formatına çevrilemedi. Geçersiz tarih girdilerini kontrol edin.")

            # Tarih farklarını hesapla
            date_diffs = (self.df[self.date_column] - self.df[self.date_column].shift()).dropna()

            # Tarih farklarının en sık tekrar eden değerini bul
            diff_counts = date_diffs.value_counts()
            most_common_diff = diff_counts.idxmax()

            # Debug: most_common_diff değerini yazdır
            print(f"Most common time difference: {most_common_diff}")

            # Frekansı belirle
            if most_common_diff <= pd.Timedelta(days=2):
                inferred_freq = 365  # Günlük (yaklaşık)
            elif most_common_diff <= pd.Timedelta(days=14):
                inferred_freq = 52  # Haftalık (yaklaşık)
            elif most_common_diff <= pd.Timedelta(days=31):
                inferred_freq = 12  # Aylık (yaklaşık)
            elif most_common_diff <= pd.Timedelta(days=366):
                inferred_freq = 1  # Yıllık
            else:
                inferred_freq = None

            return inferred_freq

        except Exception as e:
            print(f"Frekans belirlenirken hata oluştu: {e}")
            return None

    def train_test_split(self, test_size=0.1):
        """
        Veriyi tarih sütununa göre eğitim ve test setlerine böler.

        :param test_size: Test verisi oranı (varsayılan 0.1)
        :return: Eğitim ve test veri çerçeveleri
        """
        self.df = self.df.sort_values(by=self.date_column)

        # Eğitim ve test bölünmesi
        split_index = int(len(self.df) * (1 - test_size))
        train = self.df.iloc[:split_index]
        test = self.df.iloc[split_index:]

        return train, test

    def calculate_metrics(self, actual, predictions):
        metrics = {}
        metrics['MAE'] = mean_absolute_error(actual, predictions)
        metrics['MSE'] = mean_squared_error(actual, predictions)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        return metrics

    def fit_exponential_smoothing(self, train, test, target_column):
        trends = [None, 'add', 'mul']
        seasonals = [None, 'add', 'mul']
        seasonal_periods_options = [2, 3, 7, 12, 30, 52, 365]

        best_score = float('inf')
        best_params = None
        best_model = None

        for trend, seasonal, seasonal_periods in product(trends, seasonals, seasonal_periods_options):
            try:
                model = ExponentialSmoothing(train[target_column], trend=trend, seasonal=seasonal,
                                             seasonal_periods=seasonal_periods).fit()
                test_predictions = model.forecast(steps=len(test))
                score = mean_absolute_error(test[target_column], test_predictions)

                if score < best_score:
                    best_score = score
                    best_params = {'trend': trend, 'seasonal': seasonal, 'seasonal_periods': seasonal_periods}
                    best_model = model

            except Exception as e:
                print(f"Exception occurred for parameters (trend={trend}, seasonal={seasonal}, seasonal_periods={seasonal_periods}): {e}")

        train_predictions = best_model.fittedvalues
        test_predictions = best_model.forecast(steps=len(test))

        train_metrics = self.calculate_metrics(train[target_column], train_predictions)
        test_metrics = self.calculate_metrics(test[target_column], test_predictions)

        return best_model, train_metrics, test_metrics, test_predictions, train_predictions

    def fit_arima(self, train, test, target_column, n_trials=12):
        def objective(trial):
            p = trial.suggest_int('p', 0, 2)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 2)

            try:
                model = ARIMA(train[target_column], order=(p, d, q)).fit()
                test_predictions = model.get_forecast(steps=len(test)).predicted_mean
                score = mean_absolute_error(test[target_column], test_predictions)
                return score
            except Exception as e:
                print(f"Exception for ARIMA({p},{d},{q}): {e}")
                return float("inf")  # Return a high score for failed trials

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print("Best Parameters for ARIMA: ", best_params)

        # Train the final model with the best parameters
        model = ARIMA(train[target_column], order=(best_params['p'], best_params['d'], best_params['q'])).fit()
        train_predictions = model.fittedvalues
        test_predictions = model.get_forecast(steps=len(test)).predicted_mean

        train_metrics = self.calculate_metrics(train[target_column], train_predictions)
        test_metrics = self.calculate_metrics(test[target_column], test_predictions)

        return model, train_metrics, test_metrics, test_predictions, train_predictions


#Çok uzun sürdüğü için train ve hiperparametre optimizasyonu çözüm bulana kadar çıkardım.
    def fit_sarimax(self, train, test, target_column, frequency, n_trials=12):
        def objective(trial):
            param = {
                'order': (
                    trial.suggest_int('p', 0, 2),
                    trial.suggest_int('d', 0, 1),
                    trial.suggest_int('q', 0, 2)
                ),
                'seasonal_order': (
                    trial.suggest_int('P', 0, 2),
                    trial.suggest_int('D', 0, 1),
                    trial.suggest_int('Q', 0, 2),
                    frequency
                )
            }

            try:
                model = SARIMAX(train[target_column], order=param['order'], seasonal_order=param['seasonal_order'])
                fit_model = model.fit(disp=False, start_params=np.zeros(model.k_params))
                test_predictions = fit_model.forecast(steps=len(test))
                metrics = self.calculate_metrics(test[target_column], test_predictions)
                return metrics['MAE']
            except Exception as e:
                print(f"Exception during model fitting: {e}")
                return float("inf")

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        best_trial = study.best_trial
        best_order = (best_trial.params['p'], best_trial.params['d'], best_trial.params['q'])
        best_seasonal_order = (best_trial.params['P'], best_trial.params['D'], best_trial.params['Q'], frequency)

        try:
            best_model = SARIMAX(train[target_column], order=best_order, seasonal_order=best_seasonal_order)
            best_model = best_model.fit(disp=False, start_params=np.zeros(best_model.k_params))
        except Exception as e:
            print(f"Exception during final model fitting: {e}")
            return None, None, None, None, None

        train_predictions = best_model.fittedvalues
        test_predictions = best_model.forecast(steps=len(test))

        train_metrics = self.calculate_metrics(train[target_column], train_predictions)
        test_metrics = self.calculate_metrics(test[target_column], test_predictions)

        return best_model, train_metrics, test_metrics, test_predictions, train_predictions

    def fit_prophet(self, train, test, target_column, n_trials=50):
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.001, 10.0),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
                'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
                'daily_seasonality': trial.suggest_categorical('daily_seasonality', [True, False]),
                'add_hourly_seasonality': trial.suggest_categorical('add_hourly_seasonality', [True, False]),
                'add_minute_seasonality': trial.suggest_categorical('add_minute_seasonality', [True, False]),
                'n_changepoints': trial.suggest_int('n_changepoints', 10, 50),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95)
            }

            train_df = train[[target_column]].reset_index().rename(columns={'index': 'ds', target_column: 'y'})
            test_df = test.reset_index().rename(columns={'index': 'ds', target_column: 'y'})

            try:
                train_df['ds'] = pd.to_datetime(train_df['ds'])
                test_df['ds'] = pd.to_datetime(test_df['ds'])
            except Exception as e:
                print(f"Date conversion error: {e}")
                return float("inf")

            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                yearly_seasonality=params['yearly_seasonality'],
                weekly_seasonality=params['weekly_seasonality'],
                daily_seasonality=params['daily_seasonality'],
                n_changepoints=params['n_changepoints'],
                changepoint_range=params['changepoint_range']
            )

            if params['add_hourly_seasonality']:
                model.add_seasonality(name='hourly', period=24, fourier_order=3)
            if params['add_minute_seasonality']:
                model.add_seasonality(name='minute', period=60, fourier_order=3)

            try:
                model.fit(train_df)
                future = test_df[['ds']].copy()
                forecast = model.predict(future)
                test_predictions = forecast['yhat'].values

                metrics = self.calculate_metrics(test_df['y'], test_predictions)
                return metrics['MAE']
            except Exception as e:
                print(f"Exception during model fitting: {e}")
                return float("inf")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params

        train_df = train[[target_column]].reset_index().rename(columns={'index': 'ds', target_column: 'y'})
        test_df = test.reset_index().rename(columns={'index': 'ds', target_column: 'y'})

        train_df['ds'] = pd.to_datetime(train_df['ds'], errors='coerce')
        test_df['ds'] = pd.to_datetime(test_df['ds'], errors='coerce')

        train_df = train_df.dropna(subset=['ds'])
        test_df = test_df.dropna(subset=['ds'])

        best_model = Prophet(
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            holidays_prior_scale=best_params['holidays_prior_scale'],
            seasonality_mode=best_params['seasonality_mode'],
            yearly_seasonality=best_params['yearly_seasonality'],
            weekly_seasonality=best_params['weekly_seasonality'],
            daily_seasonality=best_params['daily_seasonality'],
            n_changepoints=best_params['n_changepoints'],
            changepoint_range=best_params['changepoint_range']
        )

        if best_params['add_hourly_seasonality']:
            best_model.add_seasonality(name='hourly', period=24, fourier_order=3)
        if best_params['add_minute_seasonality']:
            best_model.add_seasonality(name='minute', period=60, fourier_order=3)

        best_model.fit(train_df)

        train_predictions = best_model.predict(train_df)['yhat'].values
        future = test_df[['ds']].copy()
        forecast = best_model.predict(future)
        test_predictions = forecast['yhat'].values

        train_metrics = self.calculate_metrics(train_df['y'], train_predictions)
        test_metrics = self.calculate_metrics(test_df['y'], test_predictions)

        return best_model, train_metrics, test_metrics, test_predictions, train_predictions

#Hatalı, çıktı vermiyor.
    def fit_bsts(self, train, test, target_column, frequency, n_trials=12):
        def objective(trial):
            n_seasons = trial.suggest_int('n_seasons', 1, 3)
            n_trend = trial.suggest_int('n_trend', 1, 3)
            n_holidays = trial.suggest_int('n_holidays', 1, 3)

            try:
                model = BSTS(train[target_column], n_seasons=n_seasons, n_trend=n_trend, n_holidays=n_holidays,
                             frequency=frequency)
                fit_model = model.fit()
                test_predictions = fit_model.forecast(steps=len(test))
                metrics = self.calculate_metrics(test[target_column], test_predictions)
                return metrics['MAE']
            except Exception as e:
                print(f"Exception during model fitting: {e}")
                trial.report(float("inf"), step=0)
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        except Exception as e:
            print(f"Exception during optimization: {e}")
            return None, {}, {}, [], []

        if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
            print("No completed trials.")
            return None, {}, {}, [], []

        best_trial = study.best_trial
        best_n_seasons = best_trial.params['n_seasons']
        best_n_trend = best_trial.params['n_trend']
        best_n_holidays = best_trial.params['n_holidays']

        try:
            best_model = BSTS(train[target_column], n_seasons=best_n_seasons, n_trend=best_n_trend,
                              n_holidays=best_n_holidays, frequency=frequency)
            best_model = best_model.fit()
        except Exception as e:
            print(f"Exception during final model fitting: {e}")
            return None, {}, {}, [], []

        train_predictions = best_model.fittedvalues
        test_predictions = best_model.forecast(steps=len(test))

        try:
            train_metrics = self.calculate_metrics(train[target_column], train_predictions)
            test_metrics = self.calculate_metrics(test[target_column], test_predictions)
        except Exception as e:
            print(f"Exception during metric calculation: {e}")
            return None, {}, {}, [], []

        return best_model, train_metrics, test_metrics, test_predictions, train_predictions



    def detect_seasonality(self, df, frequency):
        """
        Zaman serisi verisindeki hedef değişkende mevsimselliği tespit eder.

        :param df: Zaman serisi veri çerçevesi
        :param frequency: Verinin frekansı
        :return: Mevsimsel bileşenler
        """
        df = df.sort_index()
        target_series = df[self.target_column]
        seasonal_periods = {'D': 7, 'W': 52, 'M': 12, 'Y': 1}.get(frequency, 12)
        seasonal_components = seasonal_decompose(target_series, model='additive', period=seasonal_periods)
        seasonal_components.plot()
        plt.suptitle(f'{self.target_column} Mevsimsel Ayrışımı')
        plt.show()

        return seasonal_components

    def visualize_predictions(self, df, date_column, target_column, predictions=None, model_name='Model',
                              dataset_type='Dataset', frequency=None):
        """
        Visualizes predictions and real values on a time series plot.

        Args:
            df (pd.DataFrame): DataFrame containing real values and predictions.
            date_column (str): Name of the column containing the date information.
            target_column (str): Name of the column containing the real values.
            predictions (np.array or pd.Series, optional): Predicted values.
            model_name (str): Name of the model.
            dataset_type (str): Type of dataset (Validation Set or Test Set).
            frequency (str, optional): Frequency of the time series data.

        Returns:
            None
        """
        if date_column not in df.columns:
            raise ValueError(f"'{date_column}' column not found in DataFrame.")
        if target_column not in df.columns:
            raise ValueError(f"'{target_column}' column not found in DataFrame.")

        # Prepare the DataFrame for real values
        real_df = df[[date_column, target_column]].copy()
        real_df.columns = ['Date', 'Real Values']
        real_df = real_df.sort_values(by='Date')

        # Prepare the DataFrame for predictions if provided
        if predictions is not None:
            if len(predictions) != len(df):
                raise ValueError("Length of predictions must match the length of the DataFrame.")
            pred_df = pd.DataFrame({'Date': df[date_column], 'Predictions': predictions})
            pred_df = pred_df[pred_df['Date'].isin(real_df['Date'])]

            # Merge real values with predictions
            result_df = pd.merge(real_df, pred_df, on='Date', how='left')
        else:
            result_df = real_df.copy()

        # Create the time series plot using Plotly
        fig = go.Figure()

        # Add trace for real values
        fig.add_trace(
            go.Scatter(x=real_df['Date'], y=real_df['Real Values'], mode='lines', name='Real Values',
                       line=dict(color='blue', shape='linear')))

        # Add trace for predictions if available
        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Predictions'], mode='lines', name='Predictions',
                           line=dict(color='red', shape='linear')))

        # Set date format and tick parameters based on frequency
        if frequency is not None:
            if frequency == 'D':
                dtick = "D1"
                tickformat = '%Y-%m-%d'
            elif frequency == 'W':
                dtick = "W1"
                tickformat = '%Y-%m-%d'
            elif frequency == 'M':
                dtick = "M1"
                tickformat = '%Y-%m'
            elif frequency == 'Y':
                dtick = "M12"
                tickformat = '%Y'
            else:
                dtick = "M1"
                tickformat = '%Y-%m-%d'
        else:
            dtick = "M1"
            tickformat = '%Y-%m-%d'

        # Update layout with titles and axis settings
        fig.update_layout(
            title=f'{model_name} - {dataset_type} Real Values vs Predictions',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Legend',
            xaxis=dict(
                type='date',
                tickformat=tickformat,
                dtick=dtick,
                tickangle=45,
                tickmode='auto'
            ),
            yaxis=dict(title='Value')
        )

        # Display the DataFrame and plot in Streamlit
        with st.expander(f"{model_name} Predictions and Real Values Dataframe:"):
            st.write(result_df)

        st.plotly_chart(fig)

    def display_metrics(self, metrics, stage):
        """
        Metrikleri ekrana yazdırır.

        Args:
            metrics (dict): Metrik değerleri içeren sözlük.
            stage (str): Metriklerin aşaması (train, val, test).
        """
        st.write(f"{stage.capitalize()} Metrics:")
        st.write(f"MAE: {metrics.get('MAE', 'Metric not available')}")
        st.write(f"MSE: {metrics.get('MSE', 'Metric not available')}")
        st.write(f"RMSE: {metrics.get('RMSE', 'Metric not available')}")