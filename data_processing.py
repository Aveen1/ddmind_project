import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from io import BytesIO

def create_date_column(df):
    """Create a date column if one doesn't exist in the DataFrame."""
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    if not date_columns:
        year_cols = [col for col in df.columns if col.lower() in ['year', 'fiscal_year', 'fy']]
        
        if year_cols:
            year_col = year_cols[0]
            try:
                df['year_temp'] = pd.to_numeric(df[year_col], errors='coerce')
                current_year = datetime.now().year
                df.loc[df['year_temp'] < 1900, 'year_temp'] = np.nan
                df.loc[df['year_temp'] > current_year + 10, 'year_temp'] = np.nan
                
                if 'Month' in df.columns:
                    df['Date'] = pd.to_datetime(dict(year=df['year_temp'], month=df['Month'], day=1)) + pd.offsets.MonthEnd(0)
                else:
                    df['Date'] = pd.to_datetime(df['year_temp'].astype(int).astype(str) + '-12-31')
                
                df = df.drop('year_temp', axis=1)
                
                return df
            except Exception as e:
                st.error(f"Error creating date from year: {e}")
                return df
        else:
            st.warning("No date or year column found. Using current date for analysis.")
            df['Date'] = datetime.now().date()
            
    return df

def process_time_period(df, date_column, time_period):
    """Process the dataframe to capture all time periods."""
    try:
        #Create a copy to avoid modifying the original
        df = df.copy()
        
        #Handle different date column scenarios
        if date_column == 'Date':
            if 'Year' in df.columns and 'Month' in df.columns:
                df[date_column] = pd.to_datetime(
                    dict(year=df['Year'], month=df['Month'], day=1)
                ) + pd.offsets.MonthEnd(0)
            elif 'Year' in df.columns:
                df[date_column] = pd.to_datetime(df['Year'].astype(str) + '-12-31')
        
        df[date_column] = pd.to_datetime(df[date_column])
        
        if time_period == "Yearly":
            df['period'] = df[date_column].dt.year.astype(str)
        
        elif time_period == "Quarterly":
            df['period'] = df[date_column].dt.to_period('Q').astype(str)
        
        elif time_period == "Monthly":
            df['period'] = df[date_column].dt.to_period('M').astype(str)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing time period: {e}")
        return df

    
def to_excel_download_link(sum_df, avg_df, count_df, filename="analysis_result.xlsx"):
    """Generates a link to download the dataframes as an Excel file with separate sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sum_df.to_excel(writer, index=True, sheet_name="Sum")
        avg_df.to_excel(writer, index=True, sheet_name="Average")
        count_df.to_excel(writer, index=True, sheet_name="Count")
    excel_data = output.getvalue()
    return excel_data
