import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from functions.analyze_data import AnalyzeColumns
from functions.data_preprocessing import DataPreprocessing
from functions.data_visualization import DataVisualization
from functions.statistical_models import StatisticalMethods
from functions.ml_models import MachineLearningMethods
from functions.feature_engineering import FeatureEngineering
import numpy as np
import joblib
import streamlit_shadcn_ui as ui

# Initialize instances
preprocessor = DataPreprocessing()
visualizor = DataVisualization()
analyzer = AnalyzeColumns()
feature_enginizer = FeatureEngineering()

st.set_page_config(page_title="SeerApp", layout="centered", page_icon="seer.png")


# Main Settings
st.title('SeerApp')

# Sidebar Settings
st.sidebar.header("**What is SeerApp ?**")
st.sidebar.write(
    "SeerApp is an innovative Time Series Auto ML project that offers cutting-edge solutions in the field of data science. This application automatically extracts meaningful insights from complex time series datasets, helping users streamline their data-driven decision-making process. With its user-friendly interface and powerful algorithms, SeerApp aims to make data science accessible to everyone.")
st.sidebar.header("**Developers**")
cols = st.sidebar.columns([1, 1, 1])
with cols[0]:
    st.link_button(label="Utku Erkul", url="https://www.linkedin.com/in/utku-erkul/")
with cols[1]:
    st.link_button(label="Selin Daştan", url="https://www.linkedin.com/in/dastanselin/")
with cols[2]:
    st.link_button(label="Samet Tuna", url="https://www.linkedin.com/in/samet-tuna-b14684169/")

with st.sidebar.expander("Model Training Information in Statistical Methods"):
    st.info("The SARIMAX model within the statistical methods section can be time-consuming and resource-intensive in terms of both training time and file size if you choose to download the model. Therefore, you have the option to select and train the models individually in this section.")
# Mode selection
mode = st.selectbox("Select Mode:", ["Auto Mode", "No Code Mode"])

# Model selection
model_type = st.selectbox("Select Model Type:", ["Statistical Models", "Machine Learning Models"])

if mode == "Auto Mode":
    st.write("Auto Mode selected.")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xls', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)

            st.write("File successfully uploaded:")
            st.dataframe(st.session_state.df)

        except Exception as e:
            st.error(f"Error: {e}")

    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

        # Find and convert date column
        date_column = preprocessor.find_and_convert_date_column(df)
        if date_column:
            st.write(f"**Date column automatically found and converted: {date_column}**")
        else:
            st.write("**Date column not found or not converted.**")
            date_column = st.selectbox("**Select the date column**", df.columns)
            if date_column:
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    st.write(f"**Selected column set as date column: {date_column}**")
                except (ValueError, TypeError):
                    st.write(f"**Selected column ({date_column}) could not be set as date column.**")

        # Find data frequency
        frequency = preprocessor.find_data_frequency(df, date_column)
        if frequency:
            st.markdown(f'**<div class="custom-text">Data frequency: {frequency}</div>**', unsafe_allow_html=True)
            window_size = preprocessor.find_optimal_window_size(frequency)
            st.write("**Optimal window size:**", window_size)
        else:
            st.write("**Could not find frequency automatically.**")
            frequency = st.number_input("**Please input Data Frequency manually**", min_value=1, max_value=len(df))

        target_columns = [col for col in df.columns if col != date_column]
        target_column = st.selectbox("**Select the feature to predict:**", target_columns)
        st.write(f"**Feature to predict: {target_column}**")

        # Data Analysis and Visualization
        analysis_results, categorical_cols, numeric_cols, stats_dict = analyzer.analyze_columns(df)


        st.subheader("**Analysis Results:**")

        with st.expander("**Observations**"):
            st.markdown(f'**<div class="custom-text">{analysis_results["Observations"]}</div>**',
                        unsafe_allow_html=True)

        with st.expander("**Variables**"):
            st.markdown(f'**<div class="custom-text">{analysis_results["Variables"]}</div>**', unsafe_allow_html=True)

        with st.expander("**Categorical Columns**"):
            for cc in analysis_results['Categorical Columns']:
                st.markdown(f'**<div class="custom-text"> {cc}</div>**', unsafe_allow_html=True)

        with st.expander("**Numerical Columns**"):
            for nc in analysis_results['Numerical Columns']:
                st.markdown(f'**<div class="custom-text"> {nc}</div>**', unsafe_allow_html=True)

        with st.expander("**Datetime Columns**"):
            for dc in analysis_results['Datetime Columns']:
                st.markdown(f'**<div class="custom-text"> {dc}</div>**', unsafe_allow_html=True)

        with st.expander("**Boolean Columns**"):
            for bc in analysis_results['Boolean Columns']:
                st.write(f"**- {bc}**")

        with st.expander("**Numerical but Categorical Columns**"):
            for nbc in analysis_results['Numerical but Categorical Columns']:
                st.write(f"**- {nbc}**")

        with st.expander("**Cardinal Categorical Columns**"):
            for ccc in analysis_results['Cardinal Categorical Columns']:
                st.write(f"**- {ccc}**")

        with st.expander("**Columns With NaN Values**"):
            for col, count in analysis_results['NaN Counts'].items():
                st.write(f"**- {col}: {count} ({analysis_results['NaN Percentage'][col]:.2f}%)**")

        with st.expander("**Descriptive Statistics**", expanded=True):
            df_descriptive_stats = pd.DataFrame.from_dict(stats_dict, orient='index')
            st.write(df_descriptive_stats)

        with st.expander("**Correlation**", expanded=True):
            fig = ff.create_annotated_heatmap(
                z=analysis_results['Correlation Matrix'].values,
                x=analysis_results['Correlation Matrix'].index.tolist(),
                y=analysis_results['Correlation Matrix'].columns.tolist(),
                colorscale='Viridis',
                annotation_text=analysis_results['Correlation Matrix'].values.round(2),
                showscale=True
            )

            fig.update_layout(
                title='Correlation Matrix',
                xaxis_title='Features',
                yaxis_title='Features',
                height=800,
                width=800,
                margin=dict(l=100, r=100, t=50, b=100),
                font=dict(size=10)
            )

            st.plotly_chart(fig)

        with st.expander("**Unique Values Count in Each Feature**", expanded=True):
            st.dataframe(analysis_results['Unique Values Count'], width=800)

        st.subheader("**Data Preparing**:")
        outlier_method = 'Clamp to 0.99 Upper Limit'
        st.session_state.df = preprocessor.handle_outliers(st.session_state.df, method='clamp')

        if set(categorical_cols).intersection(st.session_state.df.columns):
            st.subheader("Handling Missing Values in Categorical Columns")
            cleaning_method_categorical = 'Fill with Most Frequent Value'
            st.session_state.df = preprocessor.handle_categorical(df=st.session_state.df, method='fill_most_frequent')
        else:
            st.write("No Categorical Columns Found")
            cleaning_method_categorical = None

        cleaning_method_numerical = 'Interpolate'
        st.session_state.df = preprocessor.handle_numerical(df=st.session_state.df, method='interpolation')

        frequency_int = preprocessor.find_data_frequency_int(df, date_column)

        # Time Series Plots and Decomposition
        with st.expander("**Time Series Plots of Features**"):
            visualizor.timeseries_plot(st.session_state.df, date_column=date_column, target_column=target_column)
            visualizor.plot_time_series_with_rolling(st.session_state.df, date_column=date_column, window=window_size)
            visualizor.ts_decompose_and_test_stationarity(st.session_state.df, date_column=date_column,
                                                          target_column_name=target_column, model="additive")


        def save_model(model, filename):
            joblib.dump(model, filename)
            return filename


        def add_download_button(filename, label):
            with open(filename, 'rb') as f:
                model_bytes = f.read()
            st.download_button(label=label, data=model_bytes, file_name=filename)


        if model_type == "Statistical Models":
            # Model selection checkboxes
            train_exp_model = st.checkbox('Train Exponential Smoothing Model', value=False)
            train_arima_model = st.checkbox('Train ARIMA Model', value=False)
            train_sarima_model = st.checkbox('Train SARIMA Model', value=False)

            if st.button("Fit Selected Models"):
                if not (train_exp_model or train_arima_model or train_sarima_model):
                    st.warning("No model selected for training")
                else:
                    st.write("Models are fitting... Please Wait")
                    st.info(
                        "This process may take a long time depending on the size of the dataset. Additionally, we are performing hyperparameter optimization to make the best predictions, which will also take some time. You must be patient to see the future."
                    )
                    stat_methods = StatisticalMethods(df, date_column, target_column, frequency)

                    if 'df' in st.session_state:
                        with st.spinner('Model fitting in progress...'):
                            progress_bar = st.progress(0)
                            # Split the dataset
                            train, test = stat_methods.train_test_split()

                            sarimax_model = None  # Initialize the model variable

                            if train_exp_model:
                                st.write("Fitting Exponential Smoothing Model...")
                                exp_model, exp_train_metrics, exp_test_metrics, exp_test_predictions, exp_train_predictions = stat_methods.fit_exponential_smoothing(
                                    train, test, target_column)

                                st.subheader("Exponential Smoothing Metrics")
                                stat_methods.display_metrics(exp_train_metrics, 'train')
                                stat_methods.display_metrics(exp_test_metrics, 'test')

                                exp_model_filename = save_model(exp_model, "exp_model.pkl")
                                add_download_button(exp_model_filename, "Download Exponential Smoothing Model")

                            if train_arima_model:
                                st.write("Fitting ARIMA Model...")
                                arima_model, arima_train_metrics, arima_test_metrics, arima_test_predictions, arima_train_predictions = stat_methods.fit_arima(
                                    train, test, target_column)

                                st.subheader("ARIMA Metrics")
                                stat_methods.display_metrics(arima_train_metrics, 'train')
                                stat_methods.display_metrics(arima_test_metrics, 'test')

                                arima_model_filename = save_model(arima_model, "arima_model.pkl")
                                add_download_button(arima_model_filename, "Download ARIMA Model")

                            if train_sarima_model:
                                st.write("Fitting SARIMA Model...")
                                try:
                                    sarimax_model, sarimax_train_metrics, sarimax_test_metrics, sarimax_test_predictions, sarimax_train_predictions = stat_methods.fit_sarimax(
                                        train, test, target_column, frequency_int)
                                    st.subheader("SARIMA Metrics")
                                    stat_methods.display_metrics(sarimax_train_metrics, 'train')
                                    stat_methods.display_metrics(sarimax_test_metrics, 'test')

                                    sarima_model_filename = save_model(sarimax_model, "sarima_model.pkl")
                                    add_download_button(sarima_model_filename, "Download SARIMA Model")

                                except Exception as e:
                                    st.error(f"An error occurred while fitting the SARIMA model: {e}")

                            progress_bar.progress(100)

        elif model_type == "Machine Learning Models":

            st.markdown("### Model With Machine Learning Methods")

            st.session_state.df = df = feature_enginizer.feature_engineering(
                df,
                date_column=date_column,
                diff_pct_columns=[target_column],
                diff_pct=range(1, 13),
                lag_columns=[target_column],
                target_column=[target_column],
                lags=range(1, 13))

            st.write("Dataframe after Feature Engineering process")
            st.dataframe(st.session_state.df)

            # Model selection checkboxes
            train_xgb_model = st.checkbox('Train XGBoost Model', value=True)
            train_rf_model = st.checkbox('Train Random Forest Model', value=True)
            train_sgd_model = st.checkbox('Train Stochastic Gradient Descent Model', value=True)

            if st.button("Train Machine Learning Models"):
                if not any([train_xgb_model, train_rf_model, train_sgd_model]):
                    st.warning("No model selected for training.")
                else:
                    # Initialize and fit machine learning models
                    ml_methods = MachineLearningMethods(st.session_state.df, date_column, target_column, frequency)

                    X_train, X_val, X_test, y_train, y_val, y_test, X_train_dates, X_val_dates, X_test_dates = ml_methods.preprocess_data(
                        df, target_column, date_column)

                    st.write("Models are fitting... Please Wait")
                    st.info(
                        "This process may take a long time depending on the size of the dataset. Additionally, we are performing hyperparameter optimization to make the best predictions, which will also take some time. You must be patient to see the future.")

                    progress_bar = st.progress(0)

                    with st.spinner('Model fitting in progress...'):
                        progress = 0

                        if train_xgb_model:
                            st.write("Fitting XGBoost Model...")
                            progress += 20
                            progress_bar.progress(progress)
                            xgb_model, xgb_metrics, xgb_val_predictions, xgb_test_predictions = ml_methods.fit_xgbm(
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                X_test,
                                y_test)
                            xgb_model_filename = save_model(xgb_model, "xgb_model.pkl")
                            add_download_button(xgb_model_filename, "Download XGBoost Model")

                        if train_rf_model:
                            st.write("Fitting Random Forest Model...")
                            progress += 30
                            progress_bar.progress(progress)
                            rf_model, rf_metrics, rf_val_predictions, rf_test_predictions = ml_methods.fit_random_forest(
                                X_train, y_train, X_val, y_val, X_test, y_test)
                            rf_model_filename = save_model(rf_model, "rf_model.pkl")
                            add_download_button(rf_model_filename, "Download Random Forest Model")

                        if train_sgd_model:
                            st.write("Fitting Stochastic Gradient Descent Model...")
                            progress += 30
                            progress_bar.progress(progress)
                            sgd_regression_model, sgd_metrics, sgd_val_predictions, sgd_test_predictions = ml_methods.fit_sgd(
                                X_train, y_train, X_val, y_val, X_test, y_test)
                            sgd_model_filename = save_model(sgd_regression_model, "sgd_model.pkl")
                            add_download_button(sgd_model_filename, "Download SGD Model")

                        st.write("Model fitting complete!")
                        progress_bar.progress(100)

                        if train_xgb_model:
                            st.subheader("XGBoost Metrics")
                            st.write("Train Metrics")
                            ml_methods.display_metrics(xgb_metrics, 'train')
                            st.write("Validation Metrics")
                            ml_methods.display_metrics(xgb_metrics, 'val')
                            st.write("Test Metrics")
                            ml_methods.display_metrics(xgb_metrics, 'test')

                        if train_rf_model:
                            st.subheader("Random Forest Metrics")
                            st.write("Train Metrics")
                            ml_methods.display_metrics(rf_metrics, 'train')
                            st.write("Validation Metrics")
                            ml_methods.display_metrics(rf_metrics, 'val')
                            st.write("Test Metrics")
                            ml_methods.display_metrics(rf_metrics, 'test')

                        if train_sgd_model:
                            st.subheader("SGD Metrics")
                            st.write("Train Metrics")
                            ml_methods.display_metrics(sgd_metrics, 'train')
                            st.write("Validation Metrics")
                            ml_methods.display_metrics(sgd_metrics, 'val')
                            st.write("Test Metrics")
                            ml_methods.display_metrics(sgd_metrics, 'test')

                        # Create validation and test dataframes
                        val_df = pd.DataFrame({'Tarih': X_val_dates, 'Gerçek Değerler': y_val})
                        test_df = pd.DataFrame({'Tarih': X_test_dates, 'Gerçek Değerler': y_test})

                        # Add predictions to dataframes
                        if train_xgb_model:
                            val_df['XGBoost Tahminleri'] = xgb_val_predictions
                            test_df['XGBoost Tahminleri'] = xgb_test_predictions

                        if train_rf_model:
                            val_df['Random Forest Tahminleri'] = rf_val_predictions
                            test_df['Random Forest Tahminleri'] = rf_test_predictions

                        if train_sgd_model:
                            val_df['SGD Tahminleri'] = sgd_val_predictions
                            test_df['SGD Tahminleri'] = sgd_test_predictions

                        # Visualize predictions
                        if train_xgb_model:
                            ml_methods.visualize_predictions(val_df, 'Tarih', 'Gerçek Değerler',
                                                             predictions=xgb_val_predictions,
                                                             model_name='XGBoost', dataset_type='Validation Set',
                                                             frequency=frequency)
                            ml_methods.visualize_predictions(test_df, 'Tarih', 'Gerçek Değerler',
                                                             predictions=xgb_test_predictions,
                                                             model_name='XGBoost', dataset_type='Test Set',
                                                             frequency=frequency)

                        if train_rf_model:
                            ml_methods.visualize_predictions(val_df, 'Tarih', 'Gerçek Değerler',
                                                             predictions=rf_val_predictions,
                                                             model_name='Random Forest', dataset_type='Validation Set',
                                                             frequency=frequency)
                            ml_methods.visualize_predictions(test_df, 'Tarih', 'Gerçek Değerler',
                                                             predictions=rf_test_predictions,
                                                             model_name='Random Forest', dataset_type='Test Set',
                                                             frequency=frequency)

                        if train_sgd_model:
                            ml_methods.visualize_predictions(val_df, 'Tarih', 'Gerçek Değerler',
                                                             predictions=sgd_val_predictions,
                                                             model_name='SGD', dataset_type='Validation Set',
                                                             frequency=frequency)
                            ml_methods.visualize_predictions(test_df, 'Tarih', 'Gerçek Değerler',
                                                             predictions=sgd_test_predictions,
                                                             model_name='SGD', dataset_type='Test Set',
                                                             frequency=frequency)


elif mode == "No Code Mode":

    st.write("No Code Mode selected.")
    st.write("This feature is under development and will be available soon.")

    st.write("Thank you for using SeerApp!")
