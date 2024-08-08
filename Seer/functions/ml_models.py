import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import plotly.express as px
import numpy as np
import optuna
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
class MachineLearningMethods:
    """
    Bu sınıf, makine öğrenmesi yöntemleriyle modelleri eğitmek ve tahminler yapmak için gerekli yöntemleri sağlar.
    """

    def __init__(self, df, date_column, target_column, frequency):
        self.scaler = StandardScaler()
        self.df = df
        self.date_column = date_column
        self.target_column = target_column
        self.frequency = frequency
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None

    def preprocess_data(self, df, target_column, date_column):
        """
        Preprocess the data for machine learning models.

        Args:
            df (pd.DataFrame): The input data frame.
            target_column (str): The name of the target column.
            date_column (str): The name of the date column.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates: Split and scaled training, validation, and test data.
        """
        # Display the initial DataFrame
        print("Initial DataFrame:")
        print(df.head())
        print(f"Columns: {df.columns.tolist()}")

        # Ensure target column and date column are not in the features
        X = df.drop(columns=[target_column, date_column])
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

        # Convert the date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        if df[date_column].isnull().any():
            raise ValueError(f"Some dates in '{date_column}' could not be parsed. Please check the date format.")

        # Reset the index for proper slicing
        df.reset_index(drop=True, inplace=True)

        # Calculate split indices
        total_size = len(df)
        train_size = int(total_size * 0.8)
        val_size = int((total_size - train_size) / 2)

        # Use iloc to handle splitting
        X_train = X.iloc[:train_size]
        X_temp = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_temp = y.iloc[train_size:]
        train_dates = dates.iloc[:train_size]
        temp_dates = dates.iloc[train_size:]

        # Split temp into validation and test
        X_val = X_temp.iloc[:val_size]
        X_test = X_temp.iloc[val_size:]
        y_val = y_temp.iloc[:val_size]
        y_test = y_temp.iloc[val_size:]
        val_dates = temp_dates.iloc[:val_size]
        test_dates = temp_dates.iloc[val_size:]

        # Check if X_train is empty after splitting
        if X_train.empty:
            raise ValueError(
                "The training feature set X_train is empty after splitting. Please check the input DataFrame.")

        # Fit the scaler on the training features and transform all sets
        print("Fitting scaler on X_train:")
        print(X_train.head())
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates

    def fit_xgbm(self, X_train, y_train, X_val, y_val, X_test, y_test):
        tscv = TimeSeriesSplit(n_splits=10)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 18),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
            }
            model = XGBRegressor(**params)
            model.set_params(early_stopping_rounds=100)
            maes = []
            for train_index, val_index in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
                preds = model.predict(X_val_cv)
                mae = mean_absolute_error(y_val_cv, preds)
                maes.append(mae)
            return np.mean(maes)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)  # Increase number of trials for better optimization

        best_params = study.best_params
        print("Best Parameters for XGBoost: ", best_params)

        # Initialize model with the best parameters and set early_stopping_rounds
        best_model = XGBRegressor(**best_params, early_stopping_rounds=100, eval_metric='mae')

        def evaluate_cross_val(X, y):
            # Ensure X and y are pandas DataFrame and Series respectively
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)

            maes, mses, rmses = [], [], []
            for train_index, val_index in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
                y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
                best_model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
                preds = best_model.predict(X_val_cv)
                mae = mean_absolute_error(y_val_cv, preds)
                mse = mean_squared_error(y_val_cv, preds)
                rmse = np.sqrt(mse)
                maes.append(mae)
                mses.append(mse)
                rmses.append(rmse)
            return np.mean(maes), np.mean(mses), np.mean(rmses)

        # Ensure X_train, X_val, X_test are DataFrame and y_train, y_val, y_test are Series
        X_train, y_train = pd.DataFrame(X_train), pd.Series(y_train)
        X_val, y_val = pd.DataFrame(X_val), pd.Series(y_val)
        X_test, y_test = pd.DataFrame(X_test), pd.Series(y_test)

        # Evaluate performance on train, validation, and test sets
        train_mae, train_mse, train_rmse = evaluate_cross_val(X_train, y_train)
        val_mae, val_mse, val_rmse = evaluate_cross_val(X_val, y_val)
        test_mae, test_mse, test_rmse = evaluate_cross_val(X_test, y_test)

        # Train the final model with early_stopping_rounds
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_predictions = best_model.predict(X_val)
        test_predictions = best_model.predict(X_test)

        metrics = {
            'train_MAE': train_mae,
            'train_MSE': train_mse,
            'train_RMSE': train_rmse,
            'val_MAE': val_mae,
            'val_MSE': val_mse,
            'val_RMSE': val_rmse,
            'test_MAE': test_mae,
            'test_MSE': test_mse,
            'test_RMSE': test_rmse
        }

        return best_model, metrics, val_predictions, test_predictions

    def fit_random_forest(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Trains a Random Forest Regressor model and evaluates it on validation and test data.
        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training target variable.
            X_val (np.array): Validation features.
            y_val (np.array): Validation target variable.
            X_test (np.array): Test features.
            y_test (np.array): Test target variable.

        Returns:
            model, metrics, val_predictions, test_predictions: Trained model, performance metrics, and predictions.
        """
        tscv = TimeSeriesSplit(n_splits=10)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 16),
                "min_samples_split": trial.suggest_int("min_samples_split", 4, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 10),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            }
            model = RandomForestRegressor(**params)
            maes = []
            for train_index, val_index in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                model.fit(X_train_cv, y_train_cv)
                preds = model.predict(X_val_cv)
                mae = mean_absolute_error(y_val_cv, preds)
                maes.append(mae)
            return np.mean(maes)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=12)

        best_params = study.best_params
        print("Best Parameters for Random Forest: ", best_params)

        best_model = RandomForestRegressor(**best_params)
        cv_scores = []

        for train_index, val_index in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
            best_model.fit(X_train_cv, y_train_cv)
            preds = best_model.predict(X_val_cv)
            mae = mean_absolute_error(y_val_cv, preds)
            cv_scores.append(mae)

        # Fit the best model on the entire training data
        best_model.fit(X_train, y_train)
        val_predictions = best_model.predict(X_val)
        test_predictions = best_model.predict(X_test)

        # Calculate performance metrics
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_rmse = np.sqrt(val_mse)

        train_predictions = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        train_rmse = np.sqrt(train_mse)

        test_mae = mean_absolute_error(y_test, test_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_rmse = np.sqrt(test_mse)

        metrics = {
            'train_MAE': train_mae,
            'train_MSE': train_mse,
            'train_RMSE': train_rmse,
            'val_MAE': val_mae,
            'val_MSE': val_mse,
            'val_RMSE': val_rmse,
            'test_MAE': test_mae,
            'test_MSE': test_mse,
            'test_RMSE': test_rmse,
            'cv_MAE': np.mean(cv_scores)
        }


        return best_model, metrics, val_predictions, test_predictions

    def fit_sgd(self, X_train, y_train, X_val, y_val, X_test, y_test):

        tscv = TimeSeriesSplit(n_splits=10)

        def objective(trial):
            params = {
                "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "optimal", "invscaling", "adaptive"]),
                "eta0": trial.suggest_float("eta0", 0.0001, 0.1, log=True),
            }
            model = SGDRegressor(**params, max_iter=10000, tol=1e-3)
            maes = []
            for train_index, val_index in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                model.fit(X_train_cv, y_train_cv)
                preds = model.predict(X_val_cv)
                mae = mean_absolute_error(y_val_cv, preds)
                maes.append(mae)
            return np.mean(maes)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        print("Best Parameters for Linear Regression: ", best_params)

        best_model = SGDRegressor(**best_params, max_iter=5000, tol=1e-3)
        cv_scores = []

        for train_index, val_index in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
            best_model.fit(X_train_cv, y_train_cv)
            preds = best_model.predict(X_val_cv)
            mae = mean_absolute_error(y_val_cv, preds)
            cv_scores.append(mae)

        best_model.fit(X_train, y_train)
        val_predictions = best_model.predict(X_val)
        test_predictions = best_model.predict(X_test)

        val_mae = mean_absolute_error(y_val, val_predictions)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_rmse = np.sqrt(val_mse)

        train_predictions = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        train_rmse = np.sqrt(train_mse)

        test_mae = mean_absolute_error(y_test, test_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_rmse = np.sqrt(test_mse)

        metrics = {
            'val_MAE': val_mae,
            'val_MSE': val_mse,
            'val_RMSE': val_rmse,
            'train_MAE': train_mae,
            'train_MSE': train_mse,
            'train_RMSE': train_rmse,
            'test_MAE': test_mae,
            'test_MSE': test_mse,
            'test_RMSE': test_rmse,
            'cv_MAE': np.mean(cv_scores)
        }

        return best_model, metrics, val_predictions, test_predictions



    def fit_and_evaluate(self, model_name):
        """
        Belirtilen modeli eğitir ve değerlendirir.
        Args:
            model_name (str): Eğitilecek modelin adı.

        Returns:
            None
        """
        # Veriyi ön işleme tabi tut
        X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates = self.preprocess_data(
            self.df, self.target_column, self.date_column)

        # Modeli seç ve eğit
        if (model_name == 'XGBoost'):
            model, metrics, val_predictions, test_predictions = self.fit_xgbm(X_train, y_train, X_val,
                                                                                          y_val, X_test, y_test)
        elif (model_name == 'Random Forest'):
            model, metrics, val_predictions, test_predictions = self.fit_random_forest(X_train, y_train,
                                                                                                   X_val, y_val, X_test,
                                                                                                   y_test)
        elif (model_name == 'SVM'):
            model, metrics, val_predictions, test_predictions = self.fit_svm(X_train, y_train, X_val, y_val,
                                                                                         X_test, y_test)
        else:
            raise ValueError("Geçersiz model adı. Lütfen 'XGBoost', 'Random Forest' veya 'SVM' seçin.")

        # Modelin performans metriklerini görüntüle
        st.write(f"Model: {model_name}")
        st.write("Validation Metrics")
        st.write("MAE :", metrics['val_MAE'])
        st.write("MSE :", metrics['val_MSE'])
        st.write("RMSE :", metrics['val_RMSE'])

        st.write("Train Metrics")
        st.write("MAE :", metrics['train_MAE'])
        st.write("MSE :", metrics['train_MSE'])
        st.write("RMSE :", metrics['train_RMSE'])

        st.write("Test Metrics")
        st.write("MAE :", metrics['test_MAE'])
        st.write("MSE :", metrics['test_MSE'])
        st.write("RMSE :", metrics['test_RMSE'])

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
        st.write(f"MAE: {metrics.get(f'{stage}_MAE', 'Metric not available')}")
        st.write(f"MSE: {metrics.get(f'{stage}_MSE', 'Metric not available')}")
        st.write(f"RMSE: {metrics.get(f'{stage}_RMSE', 'Metric not available')}")