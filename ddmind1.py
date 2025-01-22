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
    """Extract columns from an uploaded Excel file."""
    try:
        df = pd.read_excel(file)
        return df.iloc[:, :num_columns]  # Select the first 'num_columns' columns
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None

def preprocess_data(df):
    """Preprocess and clean the extracted data."""
    try:
        steps = {
            "Remove duplicate rows": lambda df: df.drop_duplicates(),
            "Remove null values": lambda df: df.dropna(),
            "Handle date columns": lambda df: handle_date_columns(df),
            #"Standardize column names": lambda df: standardize_column_names(df),
            #"Ensure 'Year' column is valid": lambda df: validate_year_column(df),
            #"Validate and correct data types": lambda df: validate_data_types(df),
            "Remove low-variance columns": lambda df: remove_low_variance_columns(df),
            #"Reset index": lambda df: df.reset_index(drop=True)
        }

        st.write("### Data Cleaning Steps:")
        for step, func in steps.items():
            if st.checkbox(step, value=True):
                df = func(df)
                #st.success(f"{step} completed.")

        st.success("All selected data cleaning steps completed successfully!")

        #Display the cleaned dataset
        st.write("### Cleaned Data:")
        st.dataframe(df)

        return df

    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None

#function for time_period
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

def validate_year_column(df):
    """Ensure 'Year' column is valid and properly formatted."""
    if 'Year' in df.columns:
        #Remove non-numeric characters and standardize format
        df['Year'] = df['Year'].astype(str).str.replace(r"[^\d]", "", regex=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
    else:
        st.error("The dataset must contain a 'Year' column.")
    return df



def handle_date_columns(df):
    """Convert and fix date formats."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df.dropna()

def standardize_column_names(df):
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def validate_data_types(df):
    """Validate and correct data types of columns."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                pass
    return df.dropna()

def remove_low_variance_columns(df):
    """Remove columns with low variance."""
    low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=low_variance_cols)

#Function for ChatGPT Recommendations
def get_chatgpt_analysis_recommendations(df):
    """Send the entire dataset to GPT-4 and get analysis recommendations."""
    prompt = (
        "List all potential analyses (don't use sterics or dash, just highlight the title) with all this information including: \n"
        "Analysis Title\n"
        "Short Description\n"
        "Recommended Filter\n"
        "Recommended Value\n"
        "Time Period to select (monthly, quarterly, yearly).\n\n"
        f"The dataset structure is:\n{df.head(5).to_string()}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        gpt_response = response["choices"][0]["message"]["content"].strip()

        #Print GPT response as plain text
        st.write("### DDMind Analysis Recommendations:")
        st.text(gpt_response)

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")

def analyze_columns(df):
    """Analyze the columns in the DataFrame."""
    return list(df.columns)

#Function for Segment Analysis
def perform_segment_analysis(df, filter_column, value_column, time_period):
    try:
        #Ensure 'Time_Period' column exists or create it
        if 'Time_Period' not in df.columns:
            if 'Year' in df.columns and 'Month' in df.columns:
                # Create 'Time_Period' as a proper datetime column
                df['Time_Period'] = pd.to_datetime(
                    df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01',
                    format='%Y-%m-%d',
                    errors='coerce'
                )
            else:
                st.error("The DataFrame does not have 'Year' and 'Month' columns to create 'Time_Period'.")
                return
        
        #Check if any dates could not be converted (resulting in NaT)
        if df['Time_Period'].isna().any():
            st.warning("Some rows have invalid 'Year' or 'Month' values. Please check your data.")
            return

        #Filter the DataFrame by the given time period
        df = filter_data_by_time_period(df, time_period)
        
        #Ensure the necessary columns exist in the filtered DataFrame
        if filter_column not in df.columns or value_column not in df.columns:
            st.error(f"Columns '{filter_column}' or '{value_column}' not found in the DataFrame.")
            return

        #Perform the grouping and aggregation
        segment_summary = (
            df.groupby(['Time_Period', filter_column])[value_column]
            .mean()
            .reset_index()
        )

        #Check if segment_summary is not empty
        if segment_summary.empty:
            st.warning(f"No data available for the specified time period: {time_period}.")
            return
        
        #Prepare data for visualization (pivot for bar_chart)
        segment_summary_pivot = segment_summary.pivot(
            index='Time_Period', columns=filter_column, values=value_column
        )

        #Display the summary table
        st.write(f"### Segment Analysis ({filter_column}):")
        st.dataframe(segment_summary)

        #Display the bar chart
        st.bar_chart(segment_summary_pivot)

    except Exception as e:
        # Display an error message if any issue occurs
        st.error(f"Error performing segment analysis: {e}")


#Function for Retention Analysis
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

#Function for Cohort Analysis
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

#Function for Regression Analysis
def perform_regression_analysis(df, independent_var, dependent_var, time_period):
    """Perform regression analysis."""
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
    st.title("Data Analysis with DDMind")
    st.write("Upload an Excel file, and we'll help you figure out the analyses that can be done based on the sample data.")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        st.write("### Extracted Data:")
        st.dataframe(df)

        #Preprocess the data
        df = preprocess_data(df)

        if df is not None:
            columns = analyze_columns(df)

            # ChatGPT Button
            if st.button("Get DDMind Analysis Recommendations"):
                st.write("Getting DDMind recommendations based on the extracted dataset...")
                get_chatgpt_analysis_recommendations(df)

            # Step 2: Ask for Independent and Dependent Variables
            st.write("### Select Variables:")
            filter_column = st.selectbox("Select Filter", columns)
            value_column = st.selectbox("Select Value", columns)

            # Step 3: Ask for Time Periods
            time_period = st.selectbox("Select Time Period:", ["Yearly", "Quarterly", "Monthly"])

            #Step 4: Generate Analysis Options
            analysis_options = ["Segment Analysis","Retention Analysis","Cohort Analysis","Regression Analysis", "Correlation Analysis"]
            analysis_choice = st.selectbox("Select Analysis Type:", analysis_options)

            # Analysis Button
            if st.button("Run Analysis"):
                if analysis_choice == "Segment Analysis":
                    perform_segment_analysis(df, filter_column, value_column,time_period)

                elif analysis_choice == "Retention Analysis":
                    perform_retention_analysis(df, filter_column, value_column, time_period)

                elif analysis_choice == "Cohort Analysis":
                    perform_cohort_analysis(df, filter_column, value_column, time_period)

                elif analysis_choice == "Regression Analysis":
                    perform_regression_analysis(df, filter_column, value_column, time_period)

                elif analysis_choice == "Correlation Analysis":
                    correlation = df[filter_column].corr(df[value_column])
                    st.write(f"Correlation: {correlation}")

if __name__ == "__main__":
    main()