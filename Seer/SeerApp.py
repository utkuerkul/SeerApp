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
import traceback
import streamlit_shadcn_ui as ui
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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


        def clear_cache():
            # Clear the cache for data and resources
            st.cache_data.clear()
            st.cache_resource.clear()
            st.stop()  # Stop the current script execution
            st.rerun()  # Rerun the app


        if model_type == "Statistical Models":
            # Model selection checkboxes
            train_exp_model = st.checkbox('Train Exponential Smoothing Model', value=True)
            train_arima_model = st.checkbox('Train ARIMA Model', value=True)
            train_sarima_model = st.checkbox('Train SARIMA Model', value=True)

            # Load existing models from session state if available
            exp_model = st.session_state.get('exp_model', None)
            arima_model = st.session_state.get('arima_model', None)
            sarimax_model = st.session_state.get('sarimax_model', None)

            if st.button("Fit Selected Models"):
                if not (train_exp_model or train_arima_model or train_sarima_model):
                    st.warning("No model selected for training")
                else:
                    st.write("Models are fitting... Please wait")
                    st.info(
                        "This process may take a long time depending on the size of the dataset. Additionally, we are performing hyperparameter optimization to make the best predictions, which will also take some time. You must be patient to see the future."
                    )
                    stat_methods = StatisticalMethods(df, date_column, target_column, frequency)

                    if 'df' in st.session_state:
                        with st.spinner('Model fitting in progress...'):
                            progress_bar = st.progress(0)
                            train, test = stat_methods.train_test_split()

                            models_fitted = False

                            if train_exp_model:
                                st.write("Fitting Exponential Smoothing Model...")
                                exp_model, exp_train_metrics, exp_test_metrics, exp_test_predictions, exp_train_predictions = stat_methods.fit_exponential_smoothing(
                                    train, test, target_column)

                                st.session_state['exp_train_metrics'] = exp_train_metrics
                                st.session_state['exp_test_metrics'] = exp_test_metrics
                                st.session_state['exp_test_predictions'] = exp_test_predictions
                                st.session_state['exp_model'] = exp_model

                                exp_model_visualization = stat_methods.visualize_predictions(
                                    df=test,
                                    date_column='Date',
                                    target_column=target_column,
                                    predictions=exp_test_predictions,
                                    model_name='Exponential Smoothing',
                                    dataset_type='Test Set',
                                    frequency=frequency
                                )

                                st.session_state['exp_model_visualization'] = exp_model_visualization
                                exp_model_filename = save_model(exp_model, "exp_model.pkl")
                                add_download_button(exp_model_filename, "Download Exponential Smoothing Model")
                                models_fitted = True

                            if train_arima_model:
                                st.write("Fitting ARIMA Model...")
                                arima_model, arima_train_metrics, arima_test_metrics, arima_test_predictions, arima_train_predictions = stat_methods.fit_arima(
                                    train, test, target_column)

                                st.session_state['arima_train_metrics'] = arima_train_metrics
                                st.session_state['arima_test_metrics'] = arima_test_metrics
                                st.session_state['arima_test_predictions'] = arima_test_predictions
                                st.session_state['arima_model'] = arima_model

                                arima_model_visualization = stat_methods.visualize_predictions(
                                    df=test,
                                    date_column='Date',
                                    target_column=target_column,
                                    predictions=arima_test_predictions,
                                    model_name='ARIMA',
                                    dataset_type='Test Set',
                                    frequency=frequency
                                )

                                st.session_state['arima_model_visualization'] = arima_model_visualization
                                arima_model_filename = save_model(arima_model, "arima_model.pkl")
                                add_download_button(arima_model_filename, "Download ARIMA Model")
                                models_fitted = True

                            if train_sarima_model:
                                st.write("Fitting SARIMA Model...")
                                try:
                                    sarimax_model, sarimax_train_metrics, sarimax_test_metrics, sarimax_test_predictions, sarimax_train_predictions = stat_methods.fit_sarimax(
                                        train, test, target_column, frequency_int)

                                    st.session_state['sarimax_train_metrics'] = sarimax_train_metrics
                                    st.session_state['sarimax_test_metrics'] = sarimax_test_metrics
                                    st.session_state['sarimax_test_predictions'] = sarimax_test_predictions
                                    st.session_state['sarimax_model'] = sarimax_model

                                    sarimax_model_visualization = stat_methods.visualize_predictions(
                                        df=test,
                                        date_column='Date',
                                        target_column=target_column,
                                        predictions=sarimax_test_predictions,
                                        model_name='SARIMA',
                                        dataset_type='Test Set',
                                        frequency=frequency
                                    )

                                    st.session_state['sarimax_model_visualization'] = sarimax_model_visualization
                                    sarima_model_filename = save_model(sarimax_model, "sarima_model.pkl")
                                    add_download_button(sarima_model_filename, "Download SARIMA Model")
                                    models_fitted = True

                                except Exception as e:
                                    st.error(f"An error occurred while fitting the SARIMA model: {e}")

                            progress_bar.progress(100)
                            st.session_state['models_fitted'] = models_fitted

            # Display metrics and visualizations if they exist in session state
            if 'exp_train_metrics' in st.session_state:
                st.subheader("Exponential Smoothing Model Metrics")
                st.write("**Train Metrics**")
                for key, value in st.session_state['exp_train_metrics'].items():
                    st.write(f"{key}: {value:.4f}")
                st.write("**Test Metrics**")
                for key, value in st.session_state['exp_test_metrics'].items():
                    st.write(f"{key}: {value:.4f}")
                st.plotly_chart(st.session_state.get('exp_model_visualization'))

            if 'arima_train_metrics' in st.session_state:
                st.subheader("ARIMA Model Metrics")
                st.write("**Train Metrics**")
                for key, value in st.session_state['arima_train_metrics'].items():
                    st.write(f"{key}: {value:.4f}")
                st.write("**Test Metrics**")
                for key, value in st.session_state['arima_test_metrics'].items():
                    st.write(f"{key}: {value:.4f}")
                st.plotly_chart(st.session_state.get('arima_model_visualization'))

            if 'sarimax_train_metrics' in st.session_state:
                st.subheader("SARIMA Model Metrics")
                st.write("**Train Metrics**")
                for key, value in st.session_state['sarimax_train_metrics'].items():
                    st.write(f"{key}: {value:.4f}")
                st.write("**Test Metrics**")
                for key, value in st.session_state['sarimax_test_metrics'].items():
                    st.write(f"{key}: {value:.4f}")
                st.plotly_chart(st.session_state.get('sarimax_model_visualization'))

            # Now, when Predict Future is pressed, it should work if models are fitted
            if 'models_fitted' in st.session_state and st.session_state['models_fitted']:
                # Future predictions
                periods = st.number_input("Enter the number of periods to forecast:", min_value=1, value=30)

                if st.button("Predict Future"):
                    predictions_df = pd.DataFrame()  # Empty DataFrame
                    last_date = df[date_column].max()
                    predictions_df['Date'] = pd.date_range(start=last_date, periods=periods + 1, freq=frequency)[1:]

                    # Exponential Smoothing model predictions
                    if 'exp_model' in st.session_state:
                        model = st.session_state['exp_model']
                        try:
                            future_df_exp = visualizor.predict_future(model=model, data=df, date_column=date_column,
                                                                      periods=periods, frequency=frequency)
                            predictions_df['Exponential Smoothing Predictions'] = future_df_exp['Predictions'].values
                        except Exception as e:
                            st.error(f"An error occurred while predicting with Exponential Smoothing model: {e}")

                    # ARIMA model predictions
                    if 'arima_model' in st.session_state:
                        model = st.session_state['arima_model']
                        try:
                            future_df_arima = visualizor.predict_future(model=model, data=df, date_column=date_column,
                                                                        periods=periods, frequency=frequency)
                            predictions_df['ARIMA Predictions'] = future_df_arima['Predictions'].values
                        except Exception as e:
                            st.error(f"An error occurred while predicting with ARIMA model: {e}")

                    # SARIMA model predictions
                    if 'sarimax_model' in st.session_state:
                        model = st.session_state['sarimax_model']
                        try:
                            future_df_sarima = visualizor.predict_future(model=model, data=df,
                                                                         date_column=date_column, periods=periods,
                                                                         frequency=frequency)
                            predictions_df['SARIMA Predictions'] = future_df_sarima['Predictions'].values
                        except Exception as e:
                            st.error(f"An error occurred while predicting with SARIMA model: {e}")

                    # Store predictions dataframe in session state
                    st.session_state['predictions_df'] = predictions_df

            # Ensure predictions are shown at the bottom after future predictions
            if 'predictions_df' in st.session_state:
                st.subheader("Future Predictions")
                st.write(st.session_state['predictions_df'])

        if model_type == "Machine Learning Models":
            st.markdown("### Model With Machine Learning Methods")

            # Initialize session state keys if they don't exist
            for key in ['xgb_model', 'rf_model', 'adaboost_model']:
                if key not in st.session_state:
                    st.session_state[key] = None

            for key in ['xgb_metrics', 'rf_metrics', 'adaboost_metrics']:
                if key not in st.session_state:
                    st.session_state[key] = {'train': {}, 'val': {}, 'test': {}}

            if 'models_fitted' not in st.session_state:
                st.session_state['models_fitted'] = False

            # Feature Engineering
            st.session_state.df = df = feature_enginizer.feature_engineering(
                df,
                date_column=date_column,
                lag_columns=[target_column],
                target_column=target_column,
                lags=range(1, 31)
            )

            st.write("Dataframe after Feature Engineering process")
            st.dataframe(st.session_state.df)

            # Model selection checkboxes
            train_xgb_model = st.checkbox('Train XGBoost Model', value=True)
            train_rf_model = st.checkbox('Train Random Forest Model', value=True)
            train_adaboost_model = st.checkbox('Train AdaBoost Model', value=True)

            if st.button("Train Machine Learning Models"):
                if not (train_xgb_model or train_rf_model or train_adaboost_model):
                    st.warning("No model selected for training.")
                else:
                    st.write("Models are fitting... Please wait")
                    st.info("This process may take a long time depending on the size of the dataset.")

                    ml_methods = MachineLearningMethods(st.session_state.df, date_column, target_column, frequency)
                    X_train, X_val, X_test, y_train, y_val, y_test, X_train_dates, X_val_dates, X_test_dates = ml_methods.preprocess_data(
                        df, target_column, date_column
                    )

                    # Save the dates for future use in session state
                    st.session_state['X_train_dates'] = X_train_dates
                    st.session_state['X_val_dates'] = X_val_dates
                    st.session_state['X_test_dates'] = X_test_dates

                    # Save the column names for future prediction use
                    st.session_state['train_columns'] = df.drop(columns=[target_column, date_column]).columns.tolist()

                    with st.spinner('Model fitting in progress...'):
                        progress_bar = st.progress(0)
                        progress = 0

                        if train_xgb_model:
                            st.write("Fitting XGBoost Model...")
                            try:
                                xgb_model, xgb_metrics, xgb_val_predictions, xgb_test_predictions = ml_methods.fit_xgbm(
                                    X_train, y_train, X_val, y_val, X_test, y_test
                                )
                                st.session_state['xgb_model'] = xgb_model
                                st.session_state['xgb_metrics']['train'] = {
                                    'MAE': xgb_metrics['train_MAE'],
                                    'MSE': xgb_metrics['train_MSE'],
                                    'RMSE': xgb_metrics['train_RMSE']
                                }
                                st.session_state['xgb_metrics']['val'] = {
                                    'MAE': xgb_metrics['val_MAE'],
                                    'MSE': xgb_metrics['val_MSE'],
                                    'RMSE': xgb_metrics['val_RMSE']
                                }
                                st.session_state['xgb_metrics']['test'] = {
                                    'MAE': xgb_metrics['test_MAE'],
                                    'MSE': xgb_metrics['test_MSE'],
                                    'RMSE': xgb_metrics['test_RMSE']
                                }
                                st.session_state['xgb_val_predictions'] = xgb_val_predictions
                                st.session_state['xgb_test_predictions'] = xgb_test_predictions

                                xgb_model_filename = save_model(xgb_model, "xgb_model.pkl")
                                add_download_button(xgb_model_filename, "Download XGBoost Model")
                            except Exception as e:
                                st.error(f"XGBoost model training failed: {e}")

                        if train_rf_model:
                            st.write("Fitting Random Forest Model...")
                            progress += 30
                            progress_bar.progress(progress)

                            try:
                                rf_model, rf_metrics, rf_val_predictions, rf_test_predictions = ml_methods.fit_random_forest(
                                    X_train, y_train, X_val, y_val, X_test, y_test
                                )
                                st.session_state['rf_model'] = rf_model
                                st.session_state['rf_metrics']['train'] = {
                                    'MAE': rf_metrics['train_MAE'],
                                    'MSE': rf_metrics['train_MSE'],
                                    'RMSE': rf_metrics['train_RMSE']
                                }
                                st.session_state['rf_metrics']['val'] = {
                                    'MAE': rf_metrics['val_MAE'],
                                    'MSE': rf_metrics['val_MSE'],
                                    'RMSE': rf_metrics['val_RMSE']
                                }
                                st.session_state['rf_metrics']['test'] = {
                                    'MAE': rf_metrics['test_MAE'],
                                    'MSE': rf_metrics['test_MSE'],
                                    'RMSE': rf_metrics['test_RMSE']
                                }
                                st.session_state['rf_val_predictions'] = rf_val_predictions
                                st.session_state['rf_test_predictions'] = rf_test_predictions

                                rf_model_filename = save_model(rf_model, "rf_model.pkl")
                                add_download_button(rf_model_filename, "Download Random Forest Model")
                            except Exception as e:
                                st.error(f"Random Forest model training failed: {e}")

                        if train_adaboost_model:
                            st.write("Fitting AdaBoost Model...")
                            progress += 30
                            progress_bar.progress(progress)

                            try:
                                adaboost_model, adaboost_metrics, adaboost_val_predictions, adaboost_test_predictions = ml_methods.fit_adaboost(
                                    X_train, y_train, X_val, y_val, X_test, y_test
                                )
                                st.session_state['adaboost_model'] = adaboost_model
                                st.session_state['adaboost_metrics']['train'] = {
                                    'MAE': adaboost_metrics['train_MAE'],
                                    'MSE': adaboost_metrics['train_MSE'],
                                    'RMSE': adaboost_metrics['train_RMSE']
                                }
                                st.session_state['adaboost_metrics']['val'] = {
                                    'MAE': adaboost_metrics['val_MAE'],
                                    'MSE': adaboost_metrics['val_MSE'],
                                    'RMSE': adaboost_metrics['val_RMSE']
                                }
                                st.session_state['adaboost_metrics']['test'] = {
                                    'MAE': adaboost_metrics['test_MAE'],
                                    'MSE': adaboost_metrics['test_MSE'],
                                    'RMSE': adaboost_metrics['test_RMSE']
                                }
                                st.session_state['adaboost_val_predictions'] = adaboost_val_predictions
                                st.session_state['adaboost_test_predictions'] = adaboost_test_predictions

                                adaboost_model_filename = save_model(adaboost_model, "adaboost_model.pkl")
                                add_download_button(adaboost_model_filename, "Download AdaBoost Model")
                            except Exception as e:
                                st.error(f"AdaBoost model training failed: {e}")

                        st.write("Model fitting complete!")
                        progress_bar.progress(100)

                        st.session_state['models_fitted'] = True

                        # Display model metrics in a formatted way
                        for model_name in ['xgb', 'rf', 'adaboost']:
                            if f'{model_name}_metrics' in st.session_state:
                                st.subheader(f"{model_name.upper()} Metrics")
                                metrics = st.session_state[f'{model_name}_metrics']
                                for phase in ['train', 'val', 'test']:
                                    if metrics.get(phase):
                                        st.write(f"**{phase.capitalize()} Metrics:**")
                                        for key, value in metrics[phase].items():
                                            st.write(f"{phase.capitalize()} {key}: {value:.4f}")
                                    else:
                                        st.write(f"No metrics available for {phase} phase.")

                        # Test seti için veri çerçevesi
                        test_df = df[df[date_column].isin(st.session_state['X_test_dates'])]

                        # Validation seti için veri çerçevesi
                        val_df = df[df[date_column].isin(st.session_state['X_val_dates'])]

                        if 'xgb_model' in st.session_state:
                            if 'xgb_val_predictions' in st.session_state:
                                st.subheader("XGBoost Model Validation Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=val_df,  # Validation seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['xgb_val_predictions'],
                                    model_name="XGBoost Model",
                                    dataset_type="Validation Set",
                                    frequency=frequency
                                )

                            if 'xgb_test_predictions' in st.session_state:
                                st.subheader("XGBoost Model Test Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=test_df,  # Test seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['xgb_test_predictions'],
                                    model_name="XGBoost Model",
                                    dataset_type="Test Set",
                                    frequency=frequency
                                )

                        if 'rf_model' in st.session_state:
                            if 'rf_val_predictions' in st.session_state:
                                st.subheader("Random Forest Model Validation Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=val_df,  # Validation seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['rf_val_predictions'],
                                    model_name="Random Forest Model",
                                    dataset_type="Validation Set",
                                    frequency=frequency
                                )

                            if 'rf_test_predictions' in st.session_state:
                                st.subheader("Random Forest Model Test Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=test_df,  # Test seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['rf_test_predictions'],
                                    model_name="Random Forest Model",
                                    dataset_type="Test Set",
                                    frequency=frequency
                                )

                        if 'adaboost_model' in st.session_state:
                            if 'adaboost_val_predictions' in st.session_state:
                                st.subheader("AdaBoost Model Validation Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=val_df,  # Validation seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['adaboost_val_predictions'],
                                    model_name="AdaBoost Model",
                                    dataset_type="Validation Set",
                                    frequency=frequency
                                )

                            if 'adaboost_test_predictions' in st.session_state:
                                st.subheader("AdaBoost Model Test Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=test_df,  # Test seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['adaboost_test_predictions'],
                                    model_name="AdaBoost Model",
                                    dataset_type="Test Set",
                                    frequency=frequency
                                )

            # Future predictions
            if st.session_state['models_fitted']:
                periods = st.number_input("Enter the number of periods to forecast:", min_value=1, value=30)

                if st.button("Predict Future"):
                    try:
                        last_date = df[date_column].max()
                        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=frequency)[1:]

                        # future_dates'in başlangıç noktası df'in son tarihinden başlayacak şekilde olmalıdır
                        prediction_df = pd.DataFrame({date_column: future_dates})
                        prediction_df[target_column] = np.nan  # Başlangıçta hedef kolonunu boş bırakıyoruz

                        # Özellik mühendisliği uygulama
                        _, prediction_df = feature_enginizer.apply_features_to_prediction_df(
                            df=df,
                            prediction_df=prediction_df,
                            date_column=date_column,
                            lag_columns=[target_column],
                            target_column=target_column,
                            lags=range(1, 31)
                        )

                        ml_methods = MachineLearningMethods(st.session_state.df, date_column, target_column,
                                                            frequency)
                        predictions_df = pd.DataFrame({date_column: prediction_df[date_column]})

                        # XGBoost Modeli için tahmin
                        if 'xgb_model' in st.session_state:
                            try:
                                train_columns = st.session_state.get('train_columns', [])
                                if not train_columns:
                                    raise ValueError(
                                        "No train columns available. Model might not have been trained properly.")

                                xgb_predictions = ml_methods.make_prediction(
                                    st.session_state['xgb_model'],
                                    target_column,
                                    date_column,
                                    prediction_df,
                                    "XGBoost",
                                    train_columns=train_columns
                                )
                                if xgb_predictions is not None:
                                    predictions_df['XGBoost Predictions'] = xgb_predictions
                            except Exception as e:
                                st.error(f"XGBoost model failed to make predictions: {e}")
                                st.error(traceback.format_exc())

                        # Random Forest Modeli için tahmin
                        if 'rf_model' in st.session_state:
                            try:
                                rf_predictions = ml_methods.make_prediction(
                                    st.session_state['rf_model'],
                                    target_column,
                                    date_column,
                                    prediction_df,
                                    "Random Forest",
                                    train_columns=st.session_state['train_columns']
                                )
                                if rf_predictions is not None:
                                    predictions_df['Random Forest Predictions'] = rf_predictions
                            except Exception as e:
                                st.error(f"Random Forest model failed to make predictions: {e}")
                                st.error(traceback.format_exc())

                        # AdaBoost Modeli için tahmin
                        if 'adaboost_model' in st.session_state:
                            try:
                                adaboost_predictions = ml_methods.make_prediction(
                                    st.session_state['adaboost_model'],
                                    target_column,
                                    date_column,
                                    prediction_df,
                                    "AdaBoost",
                                    train_columns=st.session_state['train_columns']
                                )
                                if adaboost_predictions is not None:
                                    predictions_df['AdaBoost Predictions'] = adaboost_predictions
                            except Exception as e:
                                st.error(f"AdaBoost model failed to make predictions: {e}")
                                st.error(traceback.format_exc())

                        # Save the metrics and plots so they persist after predictions
                        for model_name in ['xgb', 'rf', 'adaboost']:
                            if f'{model_name}_metrics' in st.session_state:
                                st.subheader(f"{model_name.upper()} Metrics")
                                metrics = st.session_state[f'{model_name}_metrics']
                                for phase in ['train', 'val', 'test']:
                                    if metrics.get(phase):
                                        st.write(f"**{phase.capitalize()} Metrics:**")
                                        for key, value in metrics[phase].items():
                                            st.write(f"{phase.capitalize()} {key}: {value:.4f}")
                                    else:
                                        st.write(f"No metrics available for {phase} phase.")

                        # Test seti için veri çerçevesi
                        test_df = df[df[date_column].isin(st.session_state['X_test_dates'])]

                        # Validation seti için veri çerçevesi
                        val_df = df[df[date_column].isin(st.session_state['X_val_dates'])]

                        if 'xgb_model' in st.session_state:
                            if 'xgb_val_predictions' in st.session_state:
                                st.subheader("XGBoost Model Validation Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=val_df,  # Validation seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['xgb_val_predictions'],
                                    model_name="XGBoost Model",
                                    dataset_type="Validation Set",
                                    frequency=frequency
                                )

                            if 'xgb_test_predictions' in st.session_state:
                                st.subheader("XGBoost Model Test Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=test_df,  # Test seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['xgb_test_predictions'],
                                    model_name="XGBoost Model",
                                    dataset_type="Test Set",
                                    frequency=frequency
                                )

                        if 'rf_model' in st.session_state:
                            if 'rf_val_predictions' in st.session_state:
                                st.subheader("Random Forest Model Validation Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=val_df,  # Validation seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['rf_val_predictions'],
                                    model_name="Random Forest Model",
                                    dataset_type="Validation Set",
                                    frequency=frequency
                                )

                            if 'rf_test_predictions' in st.session_state:
                                st.subheader("Random Forest Model Test Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=test_df,  # Test seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['rf_test_predictions'],
                                    model_name="Random Forest Model",
                                    dataset_type="Test Set",
                                    frequency=frequency
                                )

                        if 'adaboost_model' in st.session_state:
                            if 'adaboost_val_predictions' in st.session_state:
                                st.subheader("AdaBoost Model Validation Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=val_df,  # Validation seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['adaboost_val_predictions'],
                                    model_name="AdaBoost Model",
                                    dataset_type="Validation Set",
                                    frequency=frequency
                                )

                            if 'adaboost_test_predictions' in st.session_state:
                                st.subheader("AdaBoost Model Test Predictions")
                                ml_methods.visualize_ml_predictions(
                                    df=test_df,  # Test seti
                                    date_column=date_column,
                                    target_column=target_column,
                                    predictions=st.session_state['adaboost_test_predictions'],
                                    model_name="AdaBoost Model",
                                    dataset_type="Test Set",
                                    frequency=frequency
                                )

                        # Processed Prediction DataFrame ve Future Predictions en altta gösterilecek
                        st.subheader("Processed Prediction DataFrame")
                        st.write(prediction_df)

                        st.subheader("Future Predictions")
                        if not predictions_df.empty:
                            st.write(predictions_df)
                        else:
                            st.write("No predictions were generated.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error(traceback.format_exc())















elif mode == "No Code Mode":

    st.write("No Code Mode selected.")
    st.write("This feature is under development and will be available soon.")

    st.write("Thank you for using SeerApp!")
