import pandas as pd
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
import numpy as np
import optuna
import plotly.graph_objects as go
import traceback
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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
        # Tarih ve hedef sütunlarını çıkarın
        X = df.drop(columns=[target_column, date_column])
        y = df[target_column]
        dates = df[date_column]

        # Boş kalma durumu kontrolü
        if X.empty:
            raise ValueError("Özellik seti tamamen boş kaldı. Veri çerçevesini kontrol edin.")

        # Tarih sütununu datetime formatına dönüştürün
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        if df[date_column].isnull().any():
            raise ValueError(f"'{date_column}' sütunundaki bazı tarihler parse edilemedi. Tarih formatını kontrol edin.")

        # Veriyi eğitim, doğrulama ve test setlerine bölün
        total_size = len(df)
        train_size = int(total_size * 0.8)
        val_size = int((total_size - train_size) / 2)

        X_train = X.iloc[:train_size]
        X_temp = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_temp = y.iloc[train_size:]
        train_dates = dates.iloc[:train_size]
        temp_dates = dates.iloc[train_size:]

        X_val = X_temp.iloc[:val_size]
        X_test = X_temp.iloc[val_size:]
        y_val = y_temp.iloc[:val_size]
        y_test = y_temp.iloc[val_size:]
        val_dates = temp_dates.iloc[:val_size]
        test_dates = temp_dates.iloc[val_size:]

        # Ölçekleyiciyi eğit ve verileri ölçeklendir
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates

    def fit_xgbm(self, X_train, y_train, X_val, y_val, X_test, y_test):
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 20.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0),
            }
            model = XGBRegressor(**params)
            model.set_params(early_stopping_rounds=50)
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
        study.optimize(objective, n_trials=12)

        best_params = study.best_params
        print("Best Parameters for XGBoost: ", best_params)

        # Initialize model with the best parameters and set early_stopping_rounds
        best_model = XGBRegressor(**best_params)
        best_model.set_params(early_stopping_rounds=50)

        def evaluate_cross_val(X, y):
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
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 10, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 50),
                "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-4, 1e-1, log=True)
            }

            model = RandomForestRegressor(**params)
            maes = []
            for train_index, val_index in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                model.fit(X_train_cv, y_train_cv)
                preds = model.predict(X_val_cv)
                maes.append(mean_absolute_error(y_val_cv, preds))
            return np.mean(maes)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        print("Best Parameters for Random Forest: ", best_params)

        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        val_predictions = best_model.predict(X_val)
        test_predictions = best_model.predict(X_test)

        metrics = {
            'train_MAE': mean_absolute_error(y_train, best_model.predict(X_train)),
            'train_MSE': mean_squared_error(y_train, best_model.predict(X_train)),
            'train_RMSE': np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
            'val_MAE': mean_absolute_error(y_val, val_predictions),
            'val_MSE': mean_squared_error(y_val, val_predictions),
            'val_RMSE': np.sqrt(mean_squared_error(y_val, val_predictions)),
            'test_MAE': mean_absolute_error(y_test, test_predictions),
            'test_MSE': mean_squared_error(y_test, test_predictions),
            'test_RMSE': np.sqrt(mean_squared_error(y_test, test_predictions))
        }

        # Return the best model, metrics, validation predictions, and test predictions
        return best_model, metrics, val_predictions, test_predictions

    def fit_adaboost(self, X_train, y_train, X_val, y_val, X_test, y_test):
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.05, log=True),
                "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
            }

            model = AdaBoostRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                loss=params["loss"],
            )

            maes = []
            for train_index, val_index in tscv.split(X_train):
                X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                model.fit(X_train_cv, y_train_cv)
                preds = model.predict(X_val_cv)
                maes.append(mean_absolute_error(y_val_cv, preds))
            return np.mean(maes)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        print("Best Parameters for AdaBoost: ", best_params)

        best_model = AdaBoostRegressor(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            loss=best_params["loss"],
        )

        best_model.fit(X_train, y_train)
        val_predictions = best_model.predict(X_val)
        test_predictions = best_model.predict(X_test)

        metrics = {
            'train_MAE': mean_absolute_error(y_train, best_model.predict(X_train)),
            'train_MSE': mean_squared_error(y_train, best_model.predict(X_train)),
            'train_RMSE': np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
            'val_MAE': mean_absolute_error(y_val, val_predictions),
            'val_MSE': mean_squared_error(y_val, val_predictions),
            'val_RMSE': np.sqrt(mean_squared_error(y_val, val_predictions)),
            'test_MAE': mean_absolute_error(y_test, test_predictions),
            'test_MSE': mean_squared_error(y_test, test_predictions),
            'test_RMSE': np.sqrt(mean_squared_error(y_test, test_predictions))
        }

        # Return the best model, metrics, validation predictions, and test predictions
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
        if model_name == 'XGBoost':
            model, metrics, val_predictions, test_predictions = self.fit_xgbm(X_train, y_train, X_val, y_val, X_test,
                                                                              y_test)
        elif model_name == 'Random Forest':
            model, metrics, val_predictions, test_predictions = self.fit_random_forest(X_train, y_train, X_val, y_val,
                                                                                       X_test, y_test)
        elif model_name == 'AdaBoost':
            model, metrics, val_predictions, test_predictions = self.fit_adaboost(X_train, y_train, X_val, y_val,
                                                                                  X_test, y_test)
        else:
            raise ValueError("Geçersiz model adı. Lütfen 'XGBoost', 'Random Forest' veya 'AdaBoost' seçin.")

        # Modelin performans metriklerini görüntüle
        st.write(f"Model: {model_name}")
        st.write("Validation Metrics")
        st.write("MAE :", metrics['val']['MAE'])
        st.write("MSE :", metrics['val']['MSE'])
        st.write("RMSE :", metrics['val']['RMSE'])

        st.write("Train Metrics")
        st.write("MAE :", metrics['train']['MAE'])
        st.write("MSE :", metrics['train']['MSE'])
        st.write("RMSE :", metrics['train']['RMSE'])

        st.write("Test Metrics")
        st.write("MAE :", metrics['test']['MAE'])
        st.write("MSE :", metrics['test']['MSE'])
        st.write("RMSE :", metrics['test']['RMSE'])

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
        if predictions is not None:
            if len(predictions) != len(df):
                raise ValueError("Tahminlerin uzunluğu veri çerçevesiyle eşleşmelidir.")

        real_df = df[[date_column, target_column]].copy()
        real_df.columns = ['Date', 'Real Values']
        real_df = real_df.sort_values(by='Date')

        if predictions is not None:
            pred_df = pd.DataFrame({'Date': df[date_column], 'Predictions': predictions})
            pred_df = pred_df[pred_df['Date'].isin(real_df['Date'])]

            result_df = pd.merge(real_df, pred_df, on='Date', how='left')
        else:
            result_df = real_df.copy()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=real_df['Date'], y=real_df['Real Values'], mode='lines', name='Real Values',
                       line=dict(color='blue', shape='linear')))

        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=pred_df['Date'], y=pred_df['Predictions'], mode='lines', name='Predictions',
                           line=dict(color='red', shape='linear')))

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

        with st.expander(f"{model_name} Predictions and Real Values Dataframe:"):
            st.write(result_df)

        st.plotly_chart(fig)

    def predict_future(self, model, data, date_column, periods, frequency, model_name):
        """
        Modeli kullanarak gelecekteki dönemler için tahmin yapar.
        Args:
            model (model object): Eğitimli model.
            data (pd.DataFrame): Tahmin yapılacak veri.
            date_column (str): Tarih sütununun adı.
            periods (int): Tahmin yapılacak dönem sayısı.
            frequency (str): Zaman serisi verisinin frekansı.
            model_name (str): Modelin adı.

        Returns:
            pd.DataFrame: Gelecek dönem tahminlerini içeren veri çerçevesi.
        """
        feature_cols = [col for col in data.columns if col != date_column]

        future_df = pd.DataFrame(
            index=pd.date_range(start=data[date_column].max(), periods=periods + 1, freq=frequency)[1:],
            columns=feature_cols
        )

        for col in feature_cols:
            future_df[col] = data[col].iloc[-1]

        future_df = future_df.reset_index()
        future_df.columns = [date_column] + feature_cols

        if model_name == "XGBoost":
            future_df = future_df.drop(columns=[date_column])
            future_df.columns = [str(i) for i in range(future_df.shape[1])]
        elif model_name in ["Random Forest", "SGD"]:
            future_df = future_df.drop(columns=[date_column])

        st.write(f"Processed Prediction DataFrame for {model_name}:")
        st.dataframe(future_df)

        predictions = model.predict(future_df)

        return pd.DataFrame({
            date_column: future_df[date_column],
            'Predictions': predictions
        })

    def make_prediction(self, model, target_column, date_column, data, model_name, train_columns=None):
        try:
            if train_columns is None:
                raise ValueError(f"train_columns is None for {model_name}. Check if the model was trained properly.")

            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=train_columns)
            elif not isinstance(data, pd.DataFrame):
                raise ValueError("Data is not in the expected format (DataFrame or numpy.ndarray).")

            if model_name == "XGBoost":
                if target_column in data.columns:
                    data = data.drop(columns=[target_column])

                missing_features = [feat for feat in train_columns if feat not in data.columns]
                extra_features = [feat for feat in data.columns if feat not in train_columns]

                for feat in missing_features:
                    data[feat] = 0

                if extra_features:
                    data = data.drop(columns=extra_features)

                data = data[train_columns]
                data.columns = [str(i) for i in range(data.shape[1])]

            elif model_name in ["Random Forest", "AdaBoost"]:
                if date_column in data.columns:
                    data = data.drop(columns=[date_column])

                missing_features = [feat for feat in train_columns if feat not in data.columns]
                extra_features = [feat for feat in data.columns if feat not in train_columns]

                for feat in missing_features:
                    data[feat] = 0

                if extra_features:
                    data = data.drop(columns=extra_features)

                data = data[train_columns]
                data = data.fillna(0)

            predictions = model.predict(data)
            return predictions

        except Exception as e:
            st.error(f"An error occurred while predicting with {model_name} model: {e}")
            st.error(traceback.format_exc())
            return None

    def visualize_ml_predictions(self, df, date_column, target_column, predictions=None, model_name='Model',
                                 dataset_type='Dataset', frequency=None):
        """
        Visualizes predictions and real values on a time series plot for machine learning models.

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