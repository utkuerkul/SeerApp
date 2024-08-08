import streamlit as st
import matplotlib as plt
import numpy as np
from sklearn.preprocessing import SplineTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import datetime


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

    def random_noise(self, series, noise_level=0.1):
        """
        Bir seriye rastgele gürültü ekler.
        """
        return series * (1 + np.random.normal(0, noise_level, len(series)))

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
            df[new_column_name] = df[target_column].rolling(window).mean()
        return df
    def create_seasonality_features(self, df, date_column):
        """
        Verilen veri çerçevesine mevsimsellik özellikleri ekler.

        Bu fonksiyon, zaman sütununa (günlük, aylık, yıllık, saatlik, dakikalık veya saniyelik) göre uygun mevsimsellik özellikleri ekler.

        Argümanlar:
            df (pandas.DataFrame): Mevsimsellik özelliklerinin ekleneceği veri çerçevesi.
            date_column (str): Zaman sütununun adı.

        Returns:
            pandas.DataFrame: Orijinal veri çerçevesi, yeni eklenmiş mevsimsellik özellikleri ile birlikte.

        Eklenen Özellikler:
            - 'sin_time': Zaman sütununun sinüs dönüşümü
            - 'cos_time': Zaman sütununun kosinüs dönüşümü

        Örnek Kullanım:
            df = pd.DataFrame({
                'timestamp': ['2022-01-01 00:00:00', '2022-01-01 01:00:00', '2022-01-01 02:00:00'],
                'value': [10, 20, 30]
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = create_seasonality_features(df, 'timestamp')
        """
        df[date_column] = pd.to_datetime(df[date_column])

        # Determine the period
        if df[date_column].dt.second.nunique() > 1:
            period = 60  # Saniyelik
        elif df[date_column].dt.minute.nunique() > 1:
            period = 60 * 60  # Dakikalık
        elif df[date_column].dt.hour.nunique() > 1:
            period = 24 * 60 * 60  # Saatlik
        elif df[date_column].dt.day.nunique() > 1:
            period = 24 * 60 * 60 * 30.4375  # Günlük (ortalama ay uzunluğu)
        elif df[date_column].dt.month.nunique() > 1:
            period = 12  # Aylık
        elif df[date_column].dt.year.nunique() > 1:
            period = 1  # Yıllık
        else:
            raise ValueError("Zaman sütununun periyodu belirlenemiyor.")

        # Create a new column for seconds since epoch
        df['timestamp_seconds'] = (df[date_column] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Add seasonal features
        df['sin_time'] = np.sin(2 * np.pi * df['timestamp_seconds'] / period)
        df['cos_time'] = np.cos(2 * np.pi * df['timestamp_seconds'] / period)

        # Drop the intermediate column
        df.drop(columns=['timestamp_seconds'], inplace=True)

        return df

    def create_periodic_spline_transformer(self, period, n_splines=None, degree=3):
        """
        Verilen periyoda göre dönemsel spline dönüştürücüsü oluşturur.

        Argümanlar:
            period (float): Dönem uzunluğu.
            n_splines (int, optional): Dönemsel spline sayısı. Varsayılan olarak None belirtilirse, period değeri kullanılır.
            degree (int, optional): Spline derecesi. Varsayılan olarak 3.

        Returns:
            sklearn.preprocessing.SplineTransformer: Dönemsel spline dönüştürücüsü.

        Örnek Kullanım:
            transformer = create_periodic_spline_transformer(period=24, n_splines=5, degree=3)
        """
        if n_splines is None:
            n_splines = int(period)
        n_knots = n_splines + 1
        return SplineTransformer(
            degree=degree,
            n_knots=n_knots,
            knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
            extrapolation="periodic",
            include_bias=True)

    def create_datetime_features(self, df, date_column, frequency):
        """
        Verilen veri çerçevesine datetime özelliklerini ekler.

        Argümanlar:
            df (pandas.DataFrame): Tarih özellikleri eklenecek veri çerçevesi.
            date_column (str): Tarih bilgisini içeren sütunun adı.
            frequency (str): Verinin frekansı ('Y', 'M', 'W', 'D', 'H', 'T', 'S') gibi.

        Dönüş:
            pandas.DataFrame: Tarih özellikleri eklenmiş veri çerçevesi.
        """
        try:
            # Tarih sütununu datetime türüne dönüştürme
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Geçersiz tarihleri kaldırın
            df = df.dropna(subset=[date_column])

            # Yıl bilgilerini ekleyin
            df['year'] = df[date_column].dt.year

            # Geçerli frekansa göre diğer özellikleri ekleyin
            if frequency in ['M', 'D', 'H', 'T', 'S']:
                df['month'] = df[date_column].dt.month
                df['quarter'] = df[date_column].dt.quarter
                df['weekofyear'] = df[date_column].dt.isocalendar().week
                df['dayofmonth'] = df[date_column].dt.day

                if frequency in ['H', 'T', 'S']:
                    df['hour'] = df[date_column].dt.hour

                if frequency in ['T', 'S']:
                    df['minute'] = df[date_column].dt.minute

            # 'Y' frekansı için sadece yıl ekleyin
            if frequency == 'Y':
                # Diğer özellikler gerekmiyorsa None olarak ayarlanır
                df['month'] = None
                df['quarter'] = None
                df['weekofyear'] = None
                df['dayofmonth'] = None
                df['hour'] = None
                df['minute'] = None
        except Exception as e:
            # Hata durumunda bilgilendirme mesajı
            print(f"An error occurred: {e}")


        return df
    def diff_pct_features(self, df, columns, diff_pct):
        """
        Belirtilen sütunlar için difference ve percentile change özelliklerini üreten fonksiyon.

        Argümanlar:
            df (pandas.DataFrame): Özelliklerin ekleneceği veri çerçevesi.
            columns (list): Difference ve percentile change özelliklerinin üretileceği sütunların adları.
            diff_pct (list): Difference ve percentile change için kullanılacak değerlerin listesi.

        Dönüş:
            pandas.DataFrame: Özellikler eklenmiş veri çerçevesi.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_but_categorical = [col for col in numeric_cols if df[col].nunique() < 5]
        for col in numeric_cols:
            if col in columns and col not in numeric_but_categorical:
                for value in diff_pct:
                    df[f'diff_{col}_{value}'] = df[col].diff(value)
                    df[f'pct_change_{col}_{value}'] = df[col].pct_change(value)
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
     #Eklenebilir
    def determine_model_type(self,time_series):
        """
        Determines whether the time series follows an additive or multiplicative model.

        Parameters:
        time_series (pd.Series): The time series data.

        Returns:
        str: 'additive' or 'multiplicative'
        """
        # Plot the original time series
        plt.figure(figsize=(10, 6))
        plt.plot(time_series, label='Original')
        plt.title('Original Time Series')
        plt.legend()
        plt.show()

        # Try additive decomposition
        additive_result = seasonal_decompose(time_series, model='additive', period=12)
        additive_result.plot()
        plt.suptitle('Additive Decomposition')
        plt.show()

        # Try multiplicative decomposition
        multiplicative_result = seasonal_decompose(time_series, model='multiplicative', period=12)
        multiplicative_result.plot()
        plt.suptitle('Multiplicative Decomposition')
        plt.show()

        # Take log transformation and try additive decomposition again
        log_time_series = np.log(time_series)
        log_additive_result = seasonal_decompose(log_time_series, model='additive', period=12)
        log_additive_result.plot()
        plt.suptitle('Log-Transformed Additive Decomposition')
        plt.show()

        # Visual inspection and comparison
        model_type = input("Based on the plots, which model seems more appropriate? (additive/multiplicative): ")
        return model_type
    def find_data_frequency(self,df, date_column):
        """
        Finds the frequency of the time series data based on the differences between consecutive datetime values,
        considering year, month, day, hour, minute, and second components.

        Parameters:
        - df (DataFrame): Input dataframe containing time series data.
        - date_column (str): Name of the date column in the dataframe.

        Returns:
        - inferred_freq (str): Inferred frequency string based on the data.
        """
        try:
            # Ensure date_column is in datetime format
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Drop duplicates in date column copy to avoid bias in frequency inference
            df['date_copy'] = df[date_column]
            df_unique = df.drop_duplicates(subset=['date_copy'])

            # Check for NaT values after conversion
            if df[date_column].isna().any():
                raise ValueError(
                    "Some dates could not be converted to datetime format. Check for invalid date entries.")

            # Calculate differences between consecutive dates
            date_diffs = (df_unique[date_column] - df_unique[date_column].shift()).dropna()

            # Count the occurrences of each difference
            diff_counts = date_diffs.value_counts()

            # Find the most common difference
            most_common_diff = diff_counts.idxmax()

            df.drop(columns=["date_copy"], axis=1, inplace=True)

            # Determine the inferred frequency based on the most common difference
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

        Parameters:
        - inferred_freq (str): Inferred frequency string based on the data.

        Returns:
        - window_size (int): Optimal window size determined based on the data frequency (in days).
        """
        try:
            # Determine window size based on the inferred frequency
            if frequency in ['D', 'B']:
                window_size = 7  # Weekly window for daily/business daily frequency
            elif frequency in ['W', 'W-SUN', 'W-MON']:
                window_size = 4  # Monthly window for weekly frequency
            elif frequency in ['M', 'BM', 'MS', 'BMS']:
                window_size = 3  # Quarterly window for monthly frequency
            elif frequency in ['Q', 'QS', 'BQ', 'BQS']:
                window_size = 4  # Annual window for quarterly frequency
            elif frequency in ['A', 'AS', 'BA', 'BAS']:
                window_size = 1  # One year window for annual frequency
            else:
                window_size = None  # Frequency not recognized, prompt user to enter manually

            return window_size

        except Exception as e:
            print(f"Error occurred while determining window size: {e}")
            return None


    def feature_engineering(self, df, date_column, target_column, diff_pct_columns, diff_pct, lag_columns, lags):
        # Ensure date_column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column])

        # Determine the frequency of the time series
        frequency = self.find_data_frequency(df, date_column)

        window_size = range(1, 13)

        # Create seasonality features
        df = self.create_seasonality_features(df, date_column)

        # Create datetime features
        df = self.create_datetime_features(df, date_column, frequency)

        # Create difference and percentile change features
        df = self.diff_pct_features(df, diff_pct_columns, diff_pct)

        # Create lag features
        df = self.lag_features(df, lag_columns, lags)

        # Create rolling mean features
        df = self.rolling_mean_features(df, window_size, target_column)

        # Create rolling shift features
        df = self.rolling_shift_features(df, lags, target_column)

        # Fill NaN values with 0
        df = df.fillna(0)

        return df