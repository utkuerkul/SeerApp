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
st.sidebar.header("**SeerApp Hakkında**")
st.sidebar.write(
    "SeerApp, veri bilimi alanında yenilikçi çözümler sunan bir Time Series Auto ML projesidir. Bu uygulama, karmaşık veri kümelerinden anlamlı bilgiler çıkararak kullanıcıların veri odaklı kararlar almasına yardımcı olur. SeerApp, kullanıcı dostu arayüzü ve güçlü algoritmalarıyla, veri bilimini herkes için erişilebilir kılmayı hedefler.")
st.sidebar.header("**Geliştiriciler**")
cols = st.sidebar.columns([1, 1, 1])
with cols[0]:
    ui.avatar(
        src="https://media.licdn.com/dms/image/C4D03AQGrSobP0d6M4Q/profile-displayphoto-shrink_800_800/0/1546034008863?e=1727913600&v=beta&t=8iouJqQSFf0ohJwAqf4MMsGycicET4H83Th8khemIls")
    st.link_button(label="Utku Erkul", url="https://www.linkedin.com/in/utku-erkul/",
                   help="https://open.spotify.com/intl-tr/track/12Ue9Y3hF9DIG2OTYF4r8g?si=ff0fb31b35084ab6")
with cols[1]:
    ui.avatar(
        src="https://media.licdn.com/dms/image/D4D03AQG7bV-4Lhdwiw/profile-displayphoto-shrink_800_800/0/1718287398289?e=1727913600&v=beta&t=1Z6c2e424WNgjxzcJUVIzos9CDvWpAEjCt5ovAplTPc")
    st.link_button(label="Selin Daştan", url="https://www.linkedin.com/in/dastanselin/",
                   help="https://open.spotify.com/intl-tr/track/6GaldcTNuEzuH8OOTeLEVB?si=bd492619674a4a3e")
with cols[2]:
    ui.avatar(
        src="https://media.licdn.com/dms/image/C4D03AQF_pGFfdO2oow/profile-displayphoto-shrink_800_800/0/1589834668048?e=1727913600&v=beta&t=4zow9zXl3vbc_LJAHPkS-9cSmW9pi-TA_xa9aN9hwcU")
    st.link_button(label="Samet Tuna", url="https://www.linkedin.com/in/samet-tuna-b14684169/",
                   help="https://open.spotify.com/intl-tr/track/3vx9E57pL0lY8x8WFlTS5b?si=a835d657821c473d")


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
            if st.button("Train Statistical Models "):

                st.write("Models are fitting... Please Wait")
                st.info(
                    "This process may take a long time depending on the size of the dataset. Additionally, we are performing hyperparameter optimization to make the best predictions, which will also take some time. You must be patient to see the future.")
                stat_methods = StatisticalMethods(df, date_column, target_column, frequency)

                if 'df' in st.session_state:
                    with st.spinner('Model fitting in progress...'):
                        progress_bar = st.progress(0)
                        # Split the dataset
                        train, test = stat_methods.train_test_split()

                        # Exponential Smoothing model
                        st.write("Fitting Exponential Smoothing Model...")
                        exp_model, exp_train_metrics, exp_test_metrics, exp_test_predictions, exp_train_predictions = stat_methods.fit_exponential_smoothing(
                            train, test, target_column)

                        # Display metrics
                        st.subheader("Exponential Smoothing Metrics")
                        stat_methods.display_metrics(exp_train_metrics, 'train')
                        stat_methods.display_metrics(exp_test_metrics, 'test')

                        # Visualize predictions
                        exp_test_df = pd.DataFrame({'Date': test[date_column], 'Real Values': test[target_column]})
                        exp_test_df['Predictions'] = exp_test_predictions
                        stat_methods.visualize_predictions(exp_test_df, 'Date', 'Real Values',
                                                           predictions=exp_test_predictions,
                                                           model_name='Exponential Smoothing',
                                                           dataset_type='Test Set', frequency=frequency)

                        # Save and add download button
                        exp_model_filename = save_model(exp_model, "exp_model.pkl")
                        add_download_button(exp_model_filename, "Download Exponential Smoothing Model")

                        progress_bar.progress(10)

                        # ARIMA model
                        st.write("Fitting ARIMA Model...")
                        arima_model, arima_train_metrics, arima_test_metrics, arima_test_predictions, arima_train_predictions = stat_methods.fit_arima(
                            train, test, target_column)
                        progress_bar.progress(20)

                        # Display metrics
                        st.subheader("ARIMA Metrics")
                        stat_methods.display_metrics(arima_train_metrics, 'train')
                        stat_methods.display_metrics(arima_test_metrics, 'test')

                        # Visualize predictions
                        arima_test_df = pd.DataFrame({'Date': test[date_column], 'Real Values': test[target_column]})
                        arima_test_df['Predictions'] = arima_test_predictions
                        stat_methods.visualize_predictions(arima_test_df, 'Date', 'Real Values',
                                                           predictions=arima_test_predictions, model_name='ARIMA',
                                                           dataset_type='Test Set', frequency=frequency)

                        arima_model_filename = save_model(arima_model, "arima_model.pkl")
                        add_download_button(arima_model_filename, "Download ARIMA Model")

                        progress_bar.progress(30)

                        # Prophet model
                        st.write("Fitting Prophet Model...")
                        prophet_model, prophet_train_metrics, prophet_test_metrics, prophet_test_predictions, prophet_train_predictions = stat_methods.fit_prophet(
                            train, test, target_column, frequency_int)
                        progress_bar.progress(50)

                        # Display metrics
                        st.subheader("Prophet Metrics")
                        stat_methods.display_metrics(prophet_train_metrics, 'train')
                        stat_methods.display_metrics(prophet_test_metrics, 'test')

                        # Visualize predictions
                        prophet_test_df = pd.DataFrame({'Date': test[date_column], 'Real Values': test[target_column]})
                        prophet_test_df['Predictions'] = prophet_test_predictions
                        progress_bar.progress(70)
                        stat_methods.visualize_predictions(prophet_test_df, 'Date', 'Real Values',
                                                           predictions=prophet_test_predictions, model_name='Prophet',
                                                           dataset_type='Test Set', frequency=frequency)

                        prophet_model_filename = save_model(prophet_model, "prophet_model.pkl")
                        add_download_button(prophet_model_filename, "Download Prophet Model")
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

            if st.button("Train Machine Learning Models "):
                # Initialize and fit machine learning models
                ml_methods = MachineLearningMethods(st.session_state.df, date_column, target_column, frequency)

                X_train, X_val, X_test, y_train, y_val, y_test, X_train_dates, X_val_dates, X_test_dates = ml_methods.preprocess_data(
                    df, target_column, date_column)

                st.write("Models are fitting... Please Wait")
                st.info(
                    "This process may take a long time depending on the size of the dataset. Additionally, we are performing hyperparameter optimization to make the best predictions, which will also take some time. You must be patient to see the future.")

                progress_bar = st.progress(0)

                with st.spinner('Model fitting in progress...'):
                    st.write("Fitting XGBoost Model...")
                    progress_bar.progress(20)
                    xgb_model, xgb_metrics, xgb_val_predictions, xgb_test_predictions = ml_methods.fit_xgbm(X_train,
                                                                                                            y_train,
                                                                                                            X_val,
                                                                                                            y_val,
                                                                                                            X_test,
                                                                                                            y_test)

                    xgb_model_filename = save_model(xgb_model, "xgb_model.pkl")
                    add_download_button(xgb_model_filename, "Download XGBoost Model")

                    st.write("Fitting Random Forest Model...")
                    progress_bar.progress(50)
                    rf_model, rf_metrics, rf_val_predictions, rf_test_predictions = ml_methods.fit_random_forest(
                        X_train, y_train, X_val, y_val, X_test, y_test)

                    rf_model_filename = save_model(rf_model, "rf_model.pkl")
                    add_download_button(rf_model_filename, "Download Random Forest Model")

                    st.write("Fitting Stochastic Gradient Descent Model...")
                    progress_bar.progress(80)
                    sgd_regression_model, sgd_metrics, sgd_val_predictions, sgd_test_predictions = ml_methods.fit_linear_regression(
                        X_train, y_train, X_val, y_val, X_test, y_test)

                    sgd_model_filename = save_model(sgd_regression_model, "sgd_model.pkl")
                    add_download_button(rf_model_filename, "Download SGD Model")
                    st.write("Model fitting complete!")
                    progress_bar.progress(100)

                    st.subheader("XGBoost Metrics")
                    st.write("Train Metrics")
                    ml_methods.display_metrics(xgb_metrics, 'train')
                    st.write("Validation Metrics")
                    ml_methods.display_metrics(xgb_metrics, 'val')
                    st.write("Test Metrics")
                    ml_methods.display_metrics(xgb_metrics, 'test')

                    st.subheader("Random Forest Metrics")
                    st.write("Train Metrics")
                    ml_methods.display_metrics(rf_metrics, 'train')
                    st.write("Validation Metrics")
                    ml_methods.display_metrics(rf_metrics, 'val')
                    st.write("Test Metrics")
                    ml_methods.display_metrics(rf_metrics, 'test')

                    st.subheader("SGD Metrics")
                    st.write("Train Metrics")
                    ml_methods.display_metrics(sgd_metrics, 'train')
                    st.write("Validation Metrics")
                    ml_methods.display_metrics(sgd_metrics, 'val')
                    st.write("Test Metrics")
                    ml_methods.display_metrics(sgd_metrics, 'test')

                    # Train, validation ve test veri çerçevelerini oluşturun
                    val_df = pd.DataFrame({'Tarih': X_val_dates, 'Gerçek Değerler': y_val})
                    test_df = pd.DataFrame({'Tarih': X_test_dates, 'Gerçek Değerler': y_test})

                    # Tahminleri veri çerçevelerine ekleyin
                    val_df['XGBoost Tahminleri'] = xgb_val_predictions
                    test_df['XGBoost Tahminleri'] = xgb_test_predictions

                    val_df['Random Forest Tahminleri'] = rf_val_predictions
                    test_df['Random Forest Tahminleri'] = rf_test_predictions

                    val_df['SGD Tahminleri'] = sgd_val_predictions
                    test_df['SGD Tahminleri'] = sgd_test_predictions

                    # Tahminleri görselleştirin
                    ml_methods.visualize_predictions(val_df, 'Tarih', 'Gerçek Değerler',
                                                     predictions=xgb_val_predictions,
                                                     model_name='XGBoost', dataset_type='Validation Set',
                                                     frequency=frequency)
                    ml_methods.visualize_predictions(test_df, 'Tarih', 'Gerçek Değerler',
                                                     predictions=xgb_test_predictions,
                                                     model_name='XGBoost', dataset_type='Test Set', frequency=frequency)

                    ml_methods.visualize_predictions(val_df, 'Tarih', 'Gerçek Değerler', predictions=rf_val_predictions,
                                                     model_name='Random Forest', dataset_type='Validation Set',
                                                     frequency=frequency)
                    ml_methods.visualize_predictions(test_df, 'Tarih', 'Gerçek Değerler',
                                                     predictions=rf_test_predictions,
                                                     model_name='Random Forest', dataset_type='Test Set',
                                                     frequency=frequency)

                    ml_methods.visualize_predictions(val_df, 'Tarih', 'Gerçek Değerler',
                                                     predictions=sgd_val_predictions,
                                                     model_name='SGD', dataset_type='Validation Set',
                                                     frequency=frequency)
                    ml_methods.visualize_predictions(test_df, 'Tarih', 'Gerçek Değerler',
                                                     predictions=sgd_test_predictions,
                                                     model_name='SGD', dataset_type='Test Set', frequency=frequency)


elif mode == "No Code Mode":
    # 1. File Upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xls', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Determine file type and read
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)

            # Show the uploaded file content
            st.write("File successfully uploaded:")
            st.dataframe(st.session_state.df)

        except Exception as e:
            st.error(f"Error: {e}")

    if 'df' in st.session_state:
        df = st.session_state.df

        date_column = preprocessor.find_and_convert_date_column(df)

        if date_column:
            st.write(f"Date columns automatically found and converted: {date_column}")
        else:
            st.write("Date column not found or not converted.")

            # Manual date column selection
            date_column = st.selectbox("Select the date column", df.columns)
            if date_column:
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    st.write(f"Selected column set as date column: {date_column}")
                except (ValueError, TypeError):
                    st.write(f"Selected column ({date_column}) could not be set as date column.")

        frequency = preprocessor.find_data_frequency(df, date_column)

        if frequency:
            st.write("Data frequency:", frequency)
            window_size = preprocessor.find_optimal_window_size(frequency)
            st.write("Optimal window size:", window_size)
            df = df.asfreq(frequency)
        else:
            st.write("Could not find frequency automatically.")
            st.number_input("Please input Data Frequency manually", min_value=1, max_value=len(df))

        # Select the target feature to predict
        target_columns = [col for col in df.columns if col != date_column]
        target_column = st.selectbox("Select the feature to predict:", target_columns)

        st.write(f"Feature to predict: {target_column}")

        # Display selected target variable
        st.write("Selected Target Variable:")
        st.write(df[target_column])

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Analyze Data", "Visualize Data", "Prepare Data", "Modeling"])

        with tab1:
            st.subheader("Analyze Data")

            if 'df_cleaned' in st.session_state:
                # Perform analysis using st.session_state.df
                analysis_results, categorical_cols, numeric_cols = analyzer.analyze_columns(st.session_state.df_cleaned)

                st.write("**Analysis Results:**")

                with st.expander("Observations"):
                    st.markdown(f"**{analysis_results['Observations']}**")

                with st.expander("Variables"):
                    st.markdown(f"**{analysis_results['Variables']}**")

                with st.expander("Categorical Columns"):
                    for cc in analysis_results['Categorical Columns']:
                        st.write(f"- {cc}")

                with st.expander("Numerical Columns"):
                    for nc in analysis_results['Numerical Columns']:
                        st.write(f"- {nc}")

                with st.expander("Datetime Columns"):
                    for dc in analysis_results['Datetime Columns']:
                        st.write(f"- {dc}")

                with st.expander("Boolean Columns"):
                    for bc in analysis_results['Boolean Columns']:
                        st.write(f"- {bc}")

                with st.expander("Numerical but Categorical Columns"):
                    for nbc in analysis_results['Numerical but Categorical Columns']:
                        st.write(f"- {nbc}")

                with st.expander("Cardinal Categorical Columns"):
                    for ccc in analysis_results['Cardinal Categorical Columns']:
                        st.write(f"- {ccc}")

                with st.expander("Columns With NaN Values"):
                    for col, count in analysis_results['NaN Counts'].items():
                        st.write(f"- {col}: {count} ({analysis_results['NaN Percentage'][col]:.2f}%)")

                with st.expander("Descriptive Statistics", expanded=True):
                    # Convert data to a DataFrame
                    df_descriptive_stats = pd.DataFrame.from_dict(analysis_results['Descriptive Statistics'],
                                                                  orient='index')

                    # Display the DataFrame as a table
                    st.write(df_descriptive_stats)

                with st.expander("Correlation", expanded=True):
                    fig = ff.create_annotated_heatmap(
                        z=analysis_results['Correlation Matrix'].values,
                        x=analysis_results['Correlation Matrix'].index.tolist(),
                        y=analysis_results['Correlation Matrix'].columns.tolist(),
                        colorscale='Viridis',
                        annotation_text=analysis_results['Correlation Matrix'].values.round(2),  # Round values
                        showscale=True  # Show color scale
                    )

                    fig.update_layout(
                        title='Correlation Matrix',
                        xaxis_title='Features',
                        yaxis_title='Features',
                        height=800,
                        width=800,
                        margin=dict(l=100, r=100, t=50, b=100),  # Increased margins
                        font=dict(size=10)  # Decreased font size
                    )

                    st.plotly_chart(fig)

                with st.expander("Unique Values Count in Each Feature", expanded=True):
                    st.dataframe(analysis_results['Unique Values Count'], width=800)
            else:
                analysis_results, categorical_cols, numeric_cols = analyzer.analyze_columns(st.session_state.df)

                st.write("**Analysis Results:**")

                with st.expander("Observations"):
                    st.markdown(f"**{analysis_results['Observations']}**")

                with st.expander("Variables"):
                    st.markdown(f"**{analysis_results['Variables']}**")

                with st.expander("Categorical Columns"):
                    for cc in analysis_results['Categorical Columns']:
                        st.write(f"- {cc}")

                with st.expander("Numerical Columns"):
                    for nc in analysis_results['Numerical Columns']:
                        st.write(f"- {nc}")

                with st.expander("Datetime Columns"):
                    for dc in analysis_results['Datetime Columns']:
                        st.write(f"- {dc}")

                with st.expander("Boolean Columns"):
                    for bc in analysis_results['Boolean Columns']:
                        st.write(f"- {bc}")

                with st.expander("Numerical but Categorical Columns"):
                    for nbc in analysis_results['Numerical but Categorical Columns']:
                        st.write(f"- {nbc}")

                with st.expander("Cardinal Categorical Columns"):
                    for ccc in analysis_results['Cardinal Categorical Columns']:
                        st.write(f"- {ccc}")

                with st.expander("Columns With NaN Values"):
                    for col, count in analysis_results['NaN Counts'].items():
                        st.write(f"- {col}: {count} ({analysis_results['NaN Percentage'][col]:.2f}%)")

                with st.expander("Descriptive Statistics", expanded=True):
                    # Convert data to a DataFrame
                    df_descriptive_stats = pd.DataFrame.from_dict(analysis_results['Descriptive Statistics'],
                                                                  orient='index')

                    # Display the DataFrame as a table
                    st.write(df_descriptive_stats)

                with st.expander("Correlation", expanded=True):
                    fig = ff.create_annotated_heatmap(
                        z=analysis_results['Correlation Matrix'].values,
                        x=analysis_results['Correlation Matrix'].index.tolist(),
                        y=analysis_results['Correlation Matrix'].columns.tolist(),
                        colorscale='Viridis',
                        annotation_text=analysis_results['Correlation Matrix'].values.round(2),  # Round values
                        showscale=True  # Show color scale
                    )

                    fig.update_layout(
                        title='Correlation Matrix',
                        xaxis_title='Features',
                        yaxis_title='Features',
                        height=800,
                        width=800,
                        margin=dict(l=100, r=100, t=50, b=100),  # Increased margins
                        font=dict(size=10)  # Decreased font size
                    )

                    st.plotly_chart(fig)

                with st.expander("Unique Values Count in Each Feature", expanded=True):
                    st.dataframe(analysis_results['Unique Values Count'], width=800)

        with tab2:
            st.title("Visualize Data")

            if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
                st.title("Time Series Plots of Features")
                visualizor.timeseries_plot(st.session_state.df_clean, date_column=date_column,
                                           target_column=target_column)
                visualizor.plot_time_series_with_rolling(st.session_state.df_clean, date_column=date_column,
                                                         window=window_size)
                visualizor.ts_decompose_and_test_stationarity(st.session_state.df_clean,
                                                              target_column_name=target_column, model="additive")
            else:
                st.title("Time Series Plots of Features")
                visualizor.timeseries_plot(st.session_state.df, date_column=date_column, target_column=target_column)
                visualizor.plot_time_series_with_rolling(st.session_state.df, date_column=date_column,
                                                         window=window_size)
                visualizor.ts_decompose_and_test_stationarity(st.session_state.df, target_column_name=target_column,
                                                              model='additive')
        with tab3:
            st.title("Prepare Data")

            if 'df' in st.session_state:
                analysis_results, categorical_cols, numerical_cols = analyzer.analyze_columns(st.session_state.df)

                if set(categorical_cols).intersection(st.session_state.df.columns):
                    st.subheader("Handling Missing Values in Categorical Columns")
                    cleaning_method_categorical = st.selectbox("How would you like to fill the missing values?",
                                                               ['Fill with Constant Value',
                                                                'Fill with Most Frequent Value',
                                                                'Drop Rows with Missing Values'])
                else:
                    st.subheader("No Categorical Columns Found")
                    cleaning_method_categorical = None

                if cleaning_method_categorical == 'Fill with Constant Value':
                    fill_value_categorical = st.text_input(
                        "Enter the constant value to fill missing values in categorical columns:")

                st.subheader("Handling Missing Values in Numerical Columns")
                cleaning_method_numerical = st.selectbox("How would you like to fill the missing values?",
                                                         ['Fill with Mean Value', 'Fill with Constant Value',
                                                          'Fill with Median Value', 'Interpolate',
                                                          'Drop Rows with Missing Values'])

                if cleaning_method_numerical == 'Fill with Constant Value':
                    fill_value_numerical = st.text_input(
                        "Enter the constant value to fill missing values in numerical columns:")

                st.subheader("Handling Outliers")
                outlier_method = st.selectbox("How would you like to handle outliers?",
                                              ['Remove Outliers', 'Replace with Mean', 'Replace with Median',
                                               'Clamp to 0.99 Upper Limit', 'Clamp to 0.95 Upper Limit', 'Leave As Is'])

                if st.button("Execute"):
                    # Categorical column operations
                    if cleaning_method_categorical == 'Fill with Constant Value' and not fill_value_categorical:
                        st.error("Please enter a constant value.")
                    elif cleaning_method_categorical == 'Fill with Constant Value':
                        st.session_state.df_clean = preprocessor.handle_categorical(df=st.session_state.df,
                                                                                    method='fill_constant',
                                                                                    fill_value=fill_value_categorical)
                    elif cleaning_method_categorical == 'Fill with Most Frequent Value':
                        st.session_state.df_clean = preprocessor.handle_categorical(df=st.session_state.df,
                                                                                    method='fill_most_frequent')
                    elif cleaning_method_categorical == 'Drop Rows with Missing Values':
                        st.session_state.df_clean = preprocessor.handle_categorical(df=st.session_state.df,
                                                                                    method='drop')

                    # Numerical column operations
                    if cleaning_method_numerical == 'Fill with Constant Value' and not fill_value_numerical:
                        st.error("Please enter a constant value.")
                    elif cleaning_method_numerical == 'Fill with Constant Value':
                        st.session_state.df_clean = preprocessor.handle_numerical(df=st.session_state.df,
                                                                                  method='fill_constant',
                                                                                  fill_value=fill_value_numerical)
                    elif cleaning_method_numerical == 'Fill with Mean Value':
                        st.session_state.df_clean = preprocessor.handle_numerical(df=st.session_state.df,
                                                                                  method='fill_mean')
                    elif cleaning_method_numerical == 'Fill with Median Value':
                        st.session_state.df_clean = preprocessor.handle_numerical(df=st.session_state.df,
                                                                                  method='fill_median')
                    elif cleaning_method_numerical == 'Interpolate':
                        st.session_state.df_clean = preprocessor.handle_numerical(df=st.session_state.df,
                                                                                  method='interpolation')
                    elif cleaning_method_numerical == 'Drop Rows with Missing Values':
                        st.session_state.df_clean = preprocessor.handle_numerical(df=st.session_state.df, method='drop')

                    # Outlier handling
                    if outlier_method == 'Remove Outliers':
                        st.session_state.df_clean = preprocessor.handle_outliers(st.session_state.df_clean,
                                                                                 method='drop')
                    elif outlier_method == 'Replace with Mean':
                        st.session_state.df_clean = preprocessor.handle_outliers(st.session_state.df_clean,
                                                                                 method='replace_with_mean')
                    elif outlier_method == 'Replace with Median':
                        st.session_state.df_clean = preprocessor.handle_outliers(st.session_state.df_clean,
                                                                                 method='replace_with_median')
                    elif outlier_method == 'Clamp to 0.99 Upper Limit':
                        st.session_state.df_clean = preprocessor.handle_outliers(st.session_state.df_clean,
                                                                                 method='clamp',
                                                                                 upper_limit=0.99)
                    elif outlier_method == 'Clamp to 0.95 Upper Limit':
                        st.session_state.df_clean = preprocessor.handle_outliers(st.session_state.df_clean,
                                                                                 method='clamp',
                                                                                 upper_limit=0.95)

                    # Display updated DataFrame
                    st.write("Operation executed successfully and data saved.")
                    st.write("Updated Data Set:")
                    st.write(f"Missing Data in Processed Data Set: {st.session_state.df_clean.isna().sum().sum()}")
                    st.dataframe(st.session_state.df_clean)

        with tab4:
            st.title("Modeling")

            if 'df' in st.session_state:
                st.write("Processed DataFrame for Modeling:")
                st.dataframe(st.session_state.df)

                choosed_model = st.radio("Please Select the Modeling Method",
                                         ["Time Series Statical Models", "Machine Learning Models"])

                if st.button("Execute Modeling"):
                    if choosed_model == "Time Series Statical Models":
                        # Implement Time Series Statical Models
                        st.write("Time Series Statical Models will be implemented.")
                    elif choosed_model == "Machine Learning Models":
                        # Implement Machine Learning Models
                        st.write("Machine Learning Models will be implemented.")
                    # Add your modeling code here

