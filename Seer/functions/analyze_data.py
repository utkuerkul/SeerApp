import plotly.figure_factory as ff
import statsmodels.api as sm
import streamlit as st

class AnalyzeColumns:
    def analyze_columns(self, df):
        """
        Analyzes the types, missing values, and correlation matrix of variables in the DataFrame.

        Args:
            df (DataFrame): The DataFrame to be analyzed.

        Returns:
            tuple: A tuple containing a dictionary of various analysis results, a list of categorical columns,
                   a list of numeric columns, and a dictionary of descriptive statistics.
        """
        # Categorical columns
        categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if
                            col not in df.select_dtypes(include=['datetime']).columns]

        # Numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # Boolean columns
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()

        # Numeric but categorical columns
        numeric_but_categorical = [col for col in numeric_cols if df[col].nunique() < 10]

        # Cardinal categorical columns
        cardinal_categoricals = [col for col in categorical_cols if df[col].nunique() > 30]

        # Columns with missing values
        nan_cols = [col for col in df.columns if df[col].isna().any()]

        # Analysis of columns with missing values
        col_analysis = {
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "datetime_cols": datetime_cols,
            "bool_cols": bool_cols,
            "numeric_but_categorical": numeric_but_categorical,
            "cardinal_categoricals": cardinal_categoricals
        }
        nan_cols_groups = {col: next(group for group, cols in col_analysis.items() if col in cols) for col in nan_cols}
        nan_counts = df[nan_cols].isna().sum().to_dict()
        nan_percentage = (df[nan_cols].isna().sum() / len(df) * 100).to_dict()

        # Descriptive statistics
        descriptive_stats = df[numeric_cols + categorical_cols + bool_cols + datetime_cols].describe(percentiles=[.10, .25, .50, .75])

        # Dictionary of descriptive statistics for each column
        stats_dict = {col: descriptive_stats.loc[col].to_dict() for col in descriptive_stats.index}

        # Correlation matrix
        correlation_matrix = df[numeric_cols].corr()

        # Unique values count
        unique_values_count = df.nunique().to_dict()

        # Number of observations
        Observations = df.shape[0]

        # Number of variables
        Variables = df.shape[1]

        var = {
            "Observations": Observations,
            "Variables": Variables,
            "Categorical Columns": categorical_cols,
            "Numerical Columns": numeric_cols,
            "Datetime Columns": datetime_cols,
            "Boolean Columns": bool_cols,
            "Numerical but Categorical Columns": numeric_but_categorical,
            "Cardinal Categorical Columns": cardinal_categoricals,
            "Columns With NaN Values": nan_cols,
            "Variable Types of Columns with NaN Values": nan_cols_groups,
            "NaN Counts": nan_counts,
            "NaN Percentage": nan_percentage,
            "Unique Values Count": unique_values_count,
            "Descriptive Statistics": descriptive_stats,
            "Correlation Matrix": correlation_matrix
        }

        return var, categorical_cols, numeric_cols, stats_dict


    def is_stationary(self,target_column):

        # "HO: Non-stationary"
        # "H1: Stationary"

        p_value = sm.tsa.stattools.adfuller(target_column)[1]

        if p_value < 0.05:
            st.write(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
        else:
            st.write(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")



