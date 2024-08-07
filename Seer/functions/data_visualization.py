import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

class DataVisualization:
    def timeseries_plot(self, df, date_column, target_column):
        """
        Verilen veri çerçevesindeki hedef sütun için zaman serisi grafiği, histogram ve otokorelasyon grafiği oluşturur.

        Argümanlar:
        df (pandas.DataFrame): Zaman serisi grafiği oluşturulacak veri çerçevesi.
        date_column (str): Zaman bilgisini içeren sütunun adı.
        target_column (str): Zaman serisi grafiği oluşturulacak hedef sütunun adı.
        """

        df = df.reset_index()
        if date_column not in df.columns:
            raise ValueError(f"'{date_column}' column not found in DataFrame.")
        if target_column not in df.columns:
            raise ValueError(f"'{target_column}' column not found in DataFrame.")

        # Zaman Serisi Grafiği
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_column], y=df[target_column], mode='lines', name=target_column,
                                 line=dict(color='blue', shape='linear')))

        fig.update_layout(title=f"{target_column} Time Series",
                          xaxis_title="Date",
                          yaxis_title=target_column,
                          title_font=dict(size=20),
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False),
                          plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig)

        # Histogram
        fig_hist = px.histogram(df, x=target_column, nbins=30, title=f"{target_column} Histogram")
        st.plotly_chart(fig_hist)

    def correlation_matrix_graph(self, df):

        # Numerik sütunları seç
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        # Sayısal ama kategorik olan sütunları filtrele
        numeric_but_categorical = [col for col in numeric_columns if df[col].nunique() < 10]
        # Kategorik olmayan sayısal sütunları seç
        num_cols = [col for col in numeric_columns if col not in numeric_but_categorical]

        # Korelasyon matrisini hesapla
        corr_matrix = df[num_cols].corr()

        # Korelasyon matrisini uzun formatta dönüştür
        corr_long = corr_matrix.reset_index().melt(id_vars='index')
        corr_long.columns = ['Variable', 'Correlation With', 'Correlation Coefficient']

        # Korelasyon matrisini çizgi grafiği olarak göster
        fig = px.line(
            corr_long,
            x='Variable',
            y='Correlation Coefficient',
            color='Correlation With',
            markers=True,
            title='Correlation Line Graph'
        )

        fig.update_yaxes(title='Correlation Coefficient', range=[-1, 1])
        fig.update_xaxes(title='', tickangle=45)

        # Grafiği Streamlit ile göster
        st.plotly_chart(fig)

    def plot_time_series_with_rolling(self, df, date_column, window):
        """
        Creates a time series plot for all numeric columns in the given DataFrame
        and shows the rolling mean and standard deviation.

        Args:
            df (pd.DataFrame): The DataFrame containing the time series data.
            date_column (str): The name of the column containing date information.
            window (int): The window size for calculating rolling mean and standard deviation.
        """
        # Ensure date_column is of datetime type
        df[date_column] = pd.to_datetime(df[date_column])

        # Drop 'index' column if it exists
        df = df.reset_index(drop=True)
        if 'index' in df.columns:
            df = df.drop(columns='index')

        # Select numeric columns for plotting
        numeric_columns = df.select_dtypes(include='number').columns

        for column in numeric_columns:
            plt.figure(figsize=(16, 6))
            plt.plot(df[date_column], df[column], label='Original', color='blue')
            plt.plot(df[date_column], df[column].rolling(window=window).mean(), label='Rolling Mean', color='orange')
            plt.plot(df[date_column], df[column].rolling(window=window).std(), label='Rolling Std', color='green')
            plt.legend()
            plt.title(f'{column} Time Series with Rolling Mean and Std (Window = {window})')
            plt.xlabel('Date')
            plt.ylabel(column)
            st.pyplot(plt)

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
    def ts_decompose_and_test_stationarity(self, df, date_column, target_column_name, model="additive"):
        """
        Decomposes the time series and tests its stationarity, and identifies trends and seasonality.

        Parameters:
        - df (DataFrame): The DataFrame containing the time series data.
        - date_column (str): The name of the date column in the DataFrame.
        - target_column_name (str): The name of the time series column to decompose and test.
        - model (str): The type of decomposition ('additive' or 'multiplicative').

        Returns:
        None
        """
        df = df.reset_index()

        # Ensure the date column and target column are in the DataFrame
        if date_column not in df.columns or target_column_name not in df.columns:
            raise ValueError(f"The columns {date_column} and/or {target_column_name} are not in the DataFrame.")

        # Ensure the date column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Set date_column as index and ensure a DatetimeIndex
        df.set_index(date_column, inplace=True)

        # Perform interpolation to handle missing values in the target column
        df[target_column_name] = df[target_column_name].interpolate(method='linear')

        frequency = self.find_data_frequency
        # Automatically determine the frequency
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            pass

            return

        if inferred_freq:
            df = df.asfreq(inferred_freq)
        elif frequency:
            df = df.asfreq(frequency)

        # Ensure the index frequency is set
        df.index.freq = inferred_freq

        # Decomposition
        try:
            result = seasonal_decompose(df[target_column_name].dropna(), model=model)
        except ValueError as e:
            st.error(f"Decomposition failed: {e}")
            return

        # Plotting
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(df.index, df[target_column_name], label='Original')
        axes[0].set_title('Original Time Series')
        axes[0].legend(loc='upper left')

        if result.trend is not None:
            axes[1].plot(df.index, result.trend, label='Trend')
            axes[1].set_title('Trend Component')
            axes[1].legend(loc='upper left')

        if result.seasonal is not None:
            axes[2].plot(df.index, result.seasonal, label='Seasonal')
            axes[2].set_title('Seasonal Component')
            axes[2].legend(loc='upper left')

        if result.resid is not None:
            axes[3].plot(df.index, result.resid, label='Residual')
            axes[3].set_title('Residual Component')
            axes[3].legend(loc='upper left')

        plt.tight_layout()
        st.pyplot(fig)
        df.reset_index(inplace=True)

        # Automatic trend and seasonality detection
        trend_direction = self.detect_trend(result.trend)
        if trend_direction == 'Upward':
            st.markdown("Upward trend detected.")
        elif trend_direction == 'Downward':
            st.markdown("Downward trend detected.")
        else:
            st.markdown("No clear trend detected.")

        if np.abs(result.seasonal.mean()) > 0.00000000001:
            st.markdown("Seasonality detected.")
        else:
            st.markdown("No significant seasonality detected.")

        # Stationarity Test
        non_na_series = df[target_column_name].dropna()
        p_value = sm.tsa.adfuller(non_na_series)[1]
        if p_value < 0.05:
            st.markdown(f"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
        else:
            st.markdown(f"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    def detect_trend(self, trend_series):
        """
        Detects the direction of the trend based on the trend series.

        Parameters:
        trend_series (pd.Series): The trend component of the time series.

        Returns:
        str: 'Upward', 'Downward', or 'No clear trend'.
        """
        if np.all(trend_series.dropna() >= 0):
            return 'Upward'
        elif np.all(trend_series.dropna() <= 0):
            return 'Downward'
        else:
            return 'No clear trend'


    def detect_seasonality(self,target, freq):
        """
        Detects seasonality in a time series.

        Parameters:
        - y (pd.Series): The time series data.
        - freq (int): The frequency of the seasonality (e.g., 12 for monthly, 4 for quarterly).

        Returns:
        - bool: True if seasonality is detected, False otherwise.
        """
        # Perform seasonal decomposition
        decomposition = sm.tsa.seasonal_decompose(target, freq=freq, model='additive')

        # Check seasonality (here we use a simple threshold for demonstration)
        seasonal_component = decomposition.seasonal
        seasonal_mean = np.abs(seasonal_component.mean())

        # Adjust the threshold as needed based on your data and context
        threshold = 0.01  # Example threshold

        # Determine if seasonality is present
        if seasonal_mean > threshold:
            return True
        else:
            return False