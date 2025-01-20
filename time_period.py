import streamlit as st
import pandas as pd
import openai
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from operator import attrgetter

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_top_columns(file, num_columns):
    try:
        df = pd.read_excel(file)
        return df.iloc[:, :num_columns]  # Select the first 'num_columns' columns
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None

def preprocess_data(df):
    try:
        steps = {
            "Remove duplicate rows": lambda df: df.drop_duplicates(),
            "Remove null values": lambda df: df.dropna(),
            "Handle date columns": lambda df: handle_date_columns(df),
            "Remove low-variance columns": lambda df: remove_low_variance_columns(df),
        }

        st.write("### Data Cleaning Steps:")
        for step, func in steps.items():
            if st.checkbox(step, value=True):
                df = func(df)

        st.success("All selected data cleaning steps completed successfully!")
        st.write("### Cleaned Data:")
        st.dataframe(df)

        return df

    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None

def handle_date_columns(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df.dropna()

def remove_low_variance_columns(df):
    low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=low_variance_cols)

def filter_data_by_time_period(df, time_period):
    if time_period == "Yearly":
        df['Time_Period'] = df['Date'].dt.to_period('Y')
    elif time_period == "Quarterly":
        df['Time_Period'] = df['Date'].dt.to_period('Q')
    elif time_period == "Monthly":
        df['Time_Period'] = df['Date'].dt.to_period('M')
    else:
        raise ValueError("Invalid time period selected.")
    return df

def perform_segment_analysis(df, filter_column, value_column, time_period):
    try:
        df = filter_data_by_time_period(df, time_period)
        segment_summary = df.groupby(['Time_Period', filter_column])[value_column].mean().reset_index()
        st.write(f"### Segment Analysis ({filter_column}):")
        st.dataframe(segment_summary)
        st.bar_chart(segment_summary.set_index(['Time_Period', filter_column]))
    except Exception as e:
        st.error(f"Error performing segment analysis: {e}")

def perform_retention_analysis(df, filter_column, value_column, time_period):
    try:
        df = filter_data_by_time_period(df, time_period)
        df['cohort'] = df.groupby(value_column)['Date'].transform('min')
        df['period'] = ((df['Date'] - df['cohort']) / pd.Timedelta(days=30)).apply(lambda x: int(x))
        retention = df.pivot_table(index='cohort', columns='period', values=value_column, aggfunc='nunique').fillna(0)
        retention_percentage = retention.div(retention.iloc[:, 0], axis=0)

        st.write("### Retention Analysis:")
        st.dataframe(retention_percentage)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Retention Analysis")
        sns.heatmap(retention_percentage, annot=True, fmt=".0%", cmap="Blues", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error performing retention analysis: {e}")

def perform_cohort_analysis(df, filter_column, value_column, time_period):
    try:
        df = filter_data_by_time_period(df, time_period)
        df['cohort'] = df.groupby(value_column)['Date'].transform('min')
        df['cohort_month'] = df['cohort'].dt.to_period('M')
        df['order_month'] = df['Date'].dt.to_period('M')
        df['cohort_index'] = (df['order_month'] - df['cohort_month']).apply(attrgetter('n'))

        cohort_data = df.groupby(['cohort_month', 'cohort_index'])[value_column].nunique().reset_index()
        cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values=value_column)
        cohort_percentage = cohort_pivot.div(cohort_pivot.iloc[:, 0], axis=0)

        st.write("### Cohort Analysis:")
        st.dataframe(cohort_percentage)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Cohort Analysis")
        sns.heatmap(cohort_percentage, annot=True, fmt=".0%", cmap="Blues", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error performing cohort analysis: {e}")

def perform_regression_analysis(df, independent_var, dependent_var, time_period):
    try:
        df = filter_data_by_time_period(df, time_period)
        X = df[independent_var]
        y = df[dependent_var]

        if X.ndim == 1:
            X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        st.write("### Regression Summary:")
        st.text(model.summary())

    except Exception as e:
        st.error(f"Error performing regression analysis: {e}")

def main():
    st.title("Data Analysis with Time Period Selection")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        st.write("### Extracted Data:")
        st.dataframe(df)

        df = preprocess_data(df)

        if df is not None:
            columns = list(df.columns)
            filter_column = st.selectbox("Select Filter Column:", columns)
            value_column = st.selectbox("Select Value Column:", columns)
            time_period = st.selectbox("Select Time Period:", ["Yearly", "Quarterly", "Monthly"])
            analysis_choice = st.selectbox("Select Analysis Type:", [
                "Segment Analysis", "Retention Analysis", "Cohort Analysis", "Regression Analysis"])

            if st.button("Run Analysis"):
                if analysis_choice == "Segment Analysis":
                    perform_segment_analysis(df, filter_column, value_column, time_period)
                elif analysis_choice == "Retention Analysis":
                    perform_retention_analysis(df, filter_column, value_column, time_period)
                elif analysis_choice == "Cohort Analysis":
                    perform_cohort_analysis(df, filter_column, value_column, time_period)
                elif analysis_choice == "Regression Analysis":
                    perform_regression_analysis(df, filter_column, value_column, time_period)

if __name__ == "__main__":
    main()
