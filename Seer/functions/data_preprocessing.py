import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self):
        self.scaler = StandardScaler()

    # Handle Missing Values
    def handle_categorical(self, df, method='most_frequent', fill_value=None):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        if method == 'most_frequent':
            for column in categorical_columns:
                most_frequent = df[column].mode()[0]
                df[column].fillna(most_frequent, inplace=True)
                st.write(f"Missing values in '{column}' categorical column have been filled with the most frequent value: {most_frequent}")

        elif method == 'fill_constant':
            if fill_value is None:
                st.error("Please provide a constant value.")
            else:
                for column in categorical_columns:
                    df[column].fillna(fill_value, inplace=True)
                    st.write(f"Missing values in '{column}' categorical column have been filled with the constant value '{fill_value}'.")

        elif method == 'drop':
            df.dropna(subset=categorical_columns, inplace=True)
            st.write("Rows with missing values in categorical columns have been dropped.")

        else:
            st.error("Invalid filling method selected.")

        return df

    def handle_numerical(self, df, method='fill_mean', fill_value=None):
        numeric_columns = df.select_dtypes(include=['number']).columns

        if method == 'fill_mean':
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            st.write("Missing values in numeric columns have been filled with the mean.")
        elif method == 'fill_median':
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            st.write("Missing values in numeric columns have been filled with the median.")
        elif method == 'fill_constant':
            if fill_value is None:
                st.error("Please provide a constant value.")
            else:
                df[numeric_columns] = df[numeric_columns].fillna(fill_value)
                st.write(f"Missing values in numeric columns have been filled with the constant value '{fill_value}'.")
        elif method == 'drop':
            df.dropna(inplace=True)
            st.write("Rows with missing values in numeric columns have been dropped.")
        elif method == 'interpolation':
            df[numeric_columns] = df[numeric_columns].interpolate()
            st.write("Missing values have been filled using interpolation.")
        elif method == "ffill":
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
            st.write("Missing values have been forward filled.")
        elif method == "bfill":
            df[numeric_columns] = df[numeric_columns].fillna(method='bfill')
            st.write("Missing values have been backward filled.")
        else:
            st.error("Invalid filling method selected.")

        return df

    # Fill Missing Dates Func (problemli)
    def fill_missing_dates(self, df, date_column, freq):
        # Tarih sütununu datetime formatına çevirelim
        if date_column in df.columns:
            # Tarih sütununu indeks olarak ayarlayalım
            df = df.set_index(date_column)

            # Günlük frekansta sıralamayı kontrol edelim
            if pd.infer_freq(df.index) == freq:
                # Veri çerçevesini tarih sırasına göre sıralayalım
                df = df.sort_index()

                # Eksik tarih değerlerini dolduralım
                if not df.index.is_monotonic_increasing:
                    # Eğer tarihler ardışık değilse ffill veya bfill ile doldur
                    df = df.asfreq('D', method='ffill')  # veya 'bfill'

            return df
        else:
            st.error("Veri çerçevesinde 'Date' adında bir sütun bulunamadı.")
            return None

    # Auto Find and Convert Date Column
    def find_and_convert_date_column(self, df):
        import pandas as pd
        import numpy as np
        import re

        date_column = None
        date_pattern = re.compile(
            r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?)|(\d{4}(?:\d{2}){0,2})'
        )

        for column in df.columns:
            if column == 'Date':
                date_column = 'Date'
                if df[column].dtype != np.datetime64:
                    df[column] = pd.to_datetime(df[column])
                break
            elif df[column].dtype == np.datetime64:
                date_column = column
                df.rename(columns={date_column: 'Date'}, inplace=True)
                date_column = 'Date'
                break
            else:
                # Check the first 5 values to determine if they are dates
                sample_values = df[column].dropna().astype(str).head()
                if sample_values.apply(lambda x: bool(date_pattern.match(x))).all():
                    try:
                        df[column] = pd.to_datetime(df[column])
                        if df[column].notna().all():  # Are all values valid dates
                            if 'Date' in df.columns and df['Date'].equals(df[column]):
                                date_column = 'Date'
                            else:
                                if 'Date' in df.columns:
                                    df.drop(columns=['Date'], inplace=True)
                                df.rename(columns={column: 'Date'}, inplace=True)
                                date_column = 'Date'
                            break
                    except (ValueError, TypeError):
                        pass

        return date_column

    def handle_outliers(self, df, method='remove', upper_limit=0.99):
        numeric_columns = df.select_dtypes(include=['number']).columns

        lower_limit = 1 - upper_limit
        Q1 = df[numeric_columns].quantile(lower_limit)
        Q3 = df[numeric_columns].quantile(upper_limit)
        IQR = Q3 - Q1

        if method == 'remove':
            # Remove rows with any outliers
            mask = ~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
            df = df[mask]
            st.write("Outliers have been removed.")
        elif method == 'fill_mean':
            # Replace outliers with mean
            for col in numeric_columns:
                mean_value = df[col].mean()
                mask = ((df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3[col] + 1.5 * IQR[col])))
                df.loc[mask, col] = mean_value
            st.write("Outliers have been replaced with the mean.")
        elif method == 'fill_median':
            # Replace outliers with median
            for col in numeric_columns:
                median_value = df[col].median()
                mask = ((df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3[col] + 1.5 * IQR[col])))
                df.loc[mask, col] = median_value
            st.write("Outliers have been replaced with the median.")
        elif method == 'clamp':
            # Clamp outliers to the upper limit
            for col in numeric_columns:
                upper_bound = df[col].quantile(upper_limit)
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            st.write(f"Outliers have been clamped to the {upper_limit} upper limit.")
        else:
            st.error("Invalid outlier handling method selected.")

        return df

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

    def find_data_frequency_int(self,df, date_column):
        """
        Verinin frekansını otomatik olarak belirler ve integer olarak çıktı verir.

        :return: Frekans değeri (integer) veya None
        """
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            if df[date_column].isna().any():
                raise ValueError(
                    "Some dates could not be converted to datetime format. Check for invalid date entries.")

            # Tarih farklarını hesapla
            date_diffs = (df[date_column] - df[date_column].shift()).dropna()

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
            print(f"Error occurred while determining data frequency: {e}")
            return None

    def detect_seasonality(self,df, target_column, frequency='M'):
        """
        Detects seasonality in the target variable of a time series dataset.

        Parameters:
        - data (pd.DataFrame): Time series data.
        - target_column (str): Name of the target variable column.
        - frequency (str, optional): Expected frequency of seasonality ('D', 'W', 'M', 'Y', etc.).

        Returns:
        - seasonal_components (statsmodels.tsa.seasonal.DecomposeResult): Result object from seasonal decomposition.
        """
        # Ensure data is sorted by date or index
        df = df.sort_index()

        # Extract the target variable as a series
        target_series = df[target_column]

        # Perform seasonal decomposition
        seasonal_components = seasonal_decompose(target_series, model='additive', period=frequency)

        # Plot seasonal decomposition components
        seasonal_components.plot()
        plt.suptitle('Seasonal Decomposition of {}'.format(target_column))
        plt.show()

        return seasonal_components
    def find_optimal_window_size(self, inferred_freq):
        """
        Finds the optimal window size for time series analysis based on the data frequency.

        Parameters:
        - inferred_freq (str): Inferred frequency string based on the data.

        Returns:
        - window_size (int): Optimal window size determined based on the data frequency (in days).
        """
        try:
            # Determine window size based on the inferred frequency
            if inferred_freq in ['D', 'B']:
                window_size = 7  # Weekly window for daily/business daily frequency
            elif inferred_freq in ['W', 'W-SUN', 'W-MON']:
                window_size = 4  # Monthly window for weekly frequency
            elif inferred_freq in ['M', 'BM', 'MS', 'BMS']:
                window_size = 3  # Quarterly window for monthly frequency
            elif inferred_freq in ['Q', 'QS', 'BQ', 'BQS']:
                window_size = 4  # Annual window for quarterly frequency
            elif inferred_freq in ['A', 'AS', 'BA', 'BAS']:
                window_size = 1  # One year window for annual frequency
            else:
                window_size = None  # Frequency not recognized, prompt user to enter manually

            return window_size

        except Exception as e:
            print(f"Error occurred while determining window size: {e}")
            return None

    def detect_outliers_all_columns(self, df, threshold=1.5):
        """
        Verilen DataFrame üzerindeki tüm numerik sütunlardaki outlier'ları tespit eden fonksiyon.

        Parametreler:
        df (DataFrame): Outlier'ları tespit etmek istediğiniz veri çerçevesi.
        threshold (float): IQR eşik değeri. Varsayılan olarak 1.5 kullanılır.

        Döndürülenler:
        dict: Sütun adları ve outlier sayıları dictionary'si.
        """
        outlier_dict = {}

        for column in df.select_dtypes(include=['number']).columns:
            Q1 = df[column].quantile(0.05)
            Q3 = df[column].quantile(0.95)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
            num_outliers = len(outlier_indices)

            outlier_dict[column] = num_outliers

        return outlier_dict

    def preprocess_data_ml(self, df, target_column, date_column):
        """
        Preprocess the data for machine learning models.

        Args:
            df (pd.DataFrame): The input data frame.
            target_column (str): The name of the target column.
            date_column (str): The name of the date column.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Split and scaled training, validation, and test data.
        """
        # Display the initial DataFrame
        print("Initial DataFrame:")
        print(df.head())
        print(f"Columns: {df.columns.tolist()}")



        # Ensure target column is not in the features
        X = df.drop(columns=[target_column])
        y = df[target_column]
        dates = df[date_column]

        # Display the feature set X and target variable y
        print("Feature set X:")
        print(X.head())
        print(f"Columns: {X.columns.tolist()}")
        print("Target variable y:")
        print(y.head())

        # Check if X is empty after dropping columns
        if X.empty:
            raise ValueError("The feature set X is empty after dropping columns. Please check the input DataFrame.")

            # Split the data into training (80%) and temp (20%)
        X_train, X_temp, y_train, y_temp, train_dates, temp_dates = train_test_split(X, y, dates, test_size=0.2,
                                                                                     random_state=42)

        # Split temp into validation (10%) and test (10%)
        X_val, X_test, y_val, y_test, val_dates, test_dates = train_test_split(X_temp, y_temp, temp_dates,
                                                                               test_size=0.5, random_state=42)

        # Check if X_train is empty after splitting
        if X_train.empty:
            raise ValueError("The training feature set X_train is empty after splitting. Please check the input DataFrame.")

        # Fit the scaler on the features and transform
        print("Fitting scaler on X_train:")
        print(X_train.head())
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates
