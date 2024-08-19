import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import SplineTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


class FeatureEngineering:
    """
    Bu class'ın amacı verilen veri çerçevesinde, class'a ait olan fonksiyonları çağırıp,
    feature engineering işlemlerini otomatik olarak yapmaktır.
    """

    def rolling_shift_features(self, df, lags, target_column):
        for lag in lags:
            new_column_name = f"{target_column}_shift_lag_{lag}"
            df[new_column_name] = df[target_column].shift(lag)
        return df

    def random_noise(self, series, noise_level=0.05):
        """
        Bir seriye rastgele düşük düzeyde gürültü ekler.
        """
        return series * (0.1 + np.random.normal(0, noise_level, len(series)))

    def lag_features(self, df, lag_columns, lags):
        """
        Belirtilen sütunlar için lag özelliklerini ekler.
        """
        for col in lag_columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag) + self.random_noise(df[col])
        return df

    def rolling_mean_features(self, df, window_size, target_column):
        for window in window_size:
            new_column_name = f"{target_column}_rolling_mean_{window}"
            df[new_column_name] = df[target_column].rolling(window, min_periods=1).mean()
        return df

    def create_seasonality_features(self, df, date_column):
        """
        Verilen veri çerçevesine mevsimsellik özellikleri ve tarihle ilgili ek bilgiler ekler.
        """
        df[date_column] = pd.to_datetime(df[date_column])

        # Yıl, ay, gün ve diğer bilgileri ekleyin
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofyear'] = df[date_column].dt.dayofyear
        df['weekday'] = df[date_column].dt.weekday
        df['is_weekend'] = (df[date_column].dt.weekday >= 5).astype(int)

        # Sinüs ve kosinüs dönüşümleri ekleyin
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        return df

    def add_trend_feature(self, df, date_column):
        """
        Verilen veri çerçevesine trend özelliği ekler.
        """
        df['trend'] = np.arange(len(df))
        return df

    def add_holiday_features(self, df, date_column, country='US'):
        """
        Belirtilen tarihe göre tatil özelliklerini ekler.
        """
        df[date_column] = pd.to_datetime(df[date_column])
        cal = calendar()
        holidays = cal.holidays(start=df[date_column].min(), end=df[date_column].max())
        df['is_holiday'] = df[date_column].isin(holidays).astype(int)
        return df

    def autoregressive_features(self, df, target_column, lags):
        """
        Otoregresif özellikler ekler.
        """
        for lag in lags:
            df[f'{target_column}_ar_{lag}'] = df[target_column].shift(lag)
        return df

    def diff_pct_features(self, df, columns, diff_pct):
        """
        Belirtilen sütunlar için difference ve percentile change özelliklerini üreten fonksiyon.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_but_categorical = [col for col in numeric_cols if df[col].nunique() < 5]
        for col in numeric_cols:
            if col in columns and col not in numeric_but_categorical:
                for value in diff_pct:
                    df[f'diff_{col}_{value}'] = df[col].diff(value)
                    df[f'pct_change_{col}_{value}'] = df[col].pct_change(value)
        df.fillna(0, inplace=True)
        return df

    def ewm_features(self, df, alphas, lags, target_column, group_columns=None):
        for alpha in alphas:
            for lag in lags:
                new_column_name = f"{target_column}_ewm_{str(alpha).replace('.', '')}_lag_{lag}"
                if group_columns:
                    df[new_column_name] = df.groupby(group_columns)[target_column].transform(
                        lambda x: x.shift(lag).ewm(alpha=alpha).mean())
                else:
                    df[new_column_name] = df[target_column].shift(lag).ewm(alpha=alpha).mean()
        return df

    def find_data_frequency(self, df, date_column):
        """
        Finds the frequency of the time series data based on the differences between consecutive datetime values,
        considering year, month, day, hour, minute, and second components.
        """
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df['date_copy'] = df[date_column]
            df_unique = df.drop_duplicates(subset=['date_copy'])
            if df[date_column].isna().any():
                raise ValueError(
                    "Some dates could not be converted to datetime format. Check for invalid date entries.")
            date_diffs = (df_unique[date_column] - df_unique[date_column].shift()).dropna()
            diff_counts = date_diffs.value_counts()
            most_common_diff = diff_counts.idxmax()
            df.drop(columns=["date_copy"], axis=1, inplace=True)
            if most_common_diff <= pd.Timedelta(days=2):
                inferred_freq = 'D'  # Daily
            elif most_common_diff <= pd.Timedelta(days=14):
                inferred_freq = 'W'  # Weekly
            elif most_common_diff <= pd.Timedelta(days=31):
                inferred_freq = 'M'  # Monthly
            elif most_common_diff <= pd.Timedelta(days=366):
                inferred_freq = 'Y'  # Yearly
            else:
                inferred_freq = None
            return inferred_freq
        except Exception as e:
            print(f"Error occurred while determining data frequency: {e}")
            return None

    def find_optimal_window_size(self, frequency):
        """
        Finds the optimal window size for time series analysis based on the data frequency.
        """
        try:
            if (frequency in ['D', 'B']):
                window_size = 7  # Weekly window for daily/business daily frequency
            elif (frequency in ['W', 'W-SUN', 'W-MON']):
                window_size = 4  # Monthly window for weekly frequency
            elif (frequency in ['M', 'BM', 'MS', 'BMS']):
                window_size = 3  # Quarterly window for monthly frequency
            elif (frequency in ['Q', 'QS', 'BQ', 'BQS']):
                window_size = 4  # Annual window for quarterly frequency
            elif (frequency in ['A', 'AS', 'BA', 'BAS']):
                window_size = 1  # One year window for annual frequency
            else:
                window_size = None  # Frequency not recognized, prompt user to enter manually
            return window_size
        except Exception as e:
            print(f"Error occurred while determining window size: {e}")
            return None

    def feature_engineering(self, df, date_column, target_column, lag_columns, lags):
        # Mevsimsellik ve trend özellikleri ekleyin
        df = self.create_seasonality_features(df, date_column)
        df = self.add_trend_feature(df, date_column)
        df = self.add_holiday_features(df, date_column)

        # Otoregresif özellikler ve lag özellikleri ekleyin
        df = self.autoregressive_features(df, target_column, lags)
        df = self.lag_features(df, lag_columns, lags)

        # Rolling ve shift özelliklerini ekleyin
        df = self.rolling_shift_features(df, lags, target_column)

        # NaN değerleri sıfırla doldurun
        df = df.fillna(0)
        return df

    def apply_features_to_prediction_df(self, df, prediction_df, date_column, target_column, lag_columns, lags):
        # Öncelikle target_column'u prediction_df'de NaN olarak ayarlıyoruz
        prediction_df[target_column] = np.nan

        # prediction_df'yi df'in son gözlemleriyle birleştiriyoruz
        combined_df = pd.concat([df, prediction_df], ignore_index=True)

        # Lag ve diğer işlemleri uyguluyoruz
        combined_df = self.feature_engineering(combined_df, date_column, target_column, lag_columns, lags)

        # Sadece prediction_df kısmını ayırıyoruz
        prediction_df = combined_df.iloc[len(df):].reset_index(drop=True)

        # Lag sütunlarındaki NaN değerlerini df'deki önceki gözlemlerle dolduruyoruz
        for lag in lags:
            lag_column_name = f"{target_column}_shift_lag_{lag}"
            if lag_column_name in prediction_df.columns:
                prediction_df[lag_column_name] = df[target_column].shift(lag).iloc[-len(prediction_df):].values

        # Burada target_column'u tekrar NaN yapıyoruz, böylece yanlışlıkla dolmaması sağlanıyor
        prediction_df[target_column] = np.nan

        # Tüm değerleri 0 olan sütunları prediction_df'den çıkarıyoruz
        zero_columns = prediction_df.columns[(prediction_df == 0).all()]
        prediction_df = prediction_df.drop(zero_columns, axis=1)

        # Aynı sütunları df'den de çıkarıyoruz
        if not zero_columns.empty:
            df = df.drop(zero_columns, axis=1)

        # Güncellenmiş df ile modelleri tekrar eğitmek için df'i geri döndürüyoruz
        return df, prediction_df