# -*- coding: utf-8 -*-
"""
HR Dataset Analysis - Applied Data Analysis
This script analyzes HR data to provide insights about:
1. Department distribution
2. Age distribution
3. Employee diversity
4. Age statistics
5. Tenure analysis
6. Hiring trends
7. Employee termination analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Set plot style
plt.style.use('default')  # Using default style since 'seaborn' is not a valid style name
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load and return HR dataset"""
    url = 'https://raw.githubusercontent.com/tomekbiel/titanic/refs/heads/main/human_resources.csv'
    try:
        df = pd.read_csv(url)
        print("\nData loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess HR data"""
    try:
        # Convert Hire Date to datetime
        df['Hire Date'] = pd.to_datetime(df['Hire Date'])
        df['Hire Year'] = df['Hire Date'].dt.year
        df['Hire Month'] = df['Hire Date'].dt.month
        
        # Convert termination date
        df['Termdate'] = pd.to_datetime(df['Termdate'], errors='coerce')
        
        # Create age bins
        df['Age (bin)'] = pd.cut(df['Age'], 
                                bins=range(18, 70, 5),
                                right=False,
                                labels=[f'{i}-{i+4}' for i in range(18, 65, 5)])
        
        print("\nData preprocessing completed!")
        return df
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return df

def analyze_department_distribution(df):
    """Analyze and visualize department distribution"""
    print("\n=== Department Distribution Analysis ===")
    
    # Get department counts
    dept_counts = df.groupby('Department')['Id'].count().reset_index()
    top_dept = dept_counts.sort_values(by='Id', ascending=False).head(1)
    
    print(f"\nDepartment with most employees: {top_dept['Department'].values[0]} ({top_dept['Id'].values[0]} employees)")
    
    # Plot department distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=dept_counts.sort_values('Id'), 
               x='Department', 
               y='Id',
               palette='viridis')
    plt.title('Department Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_age_distribution(df):
    """Analyze and visualize age distribution"""
    print("\n=== Age Distribution Analysis ===")
    
    # Get age distribution
    age_counts = df.groupby('Age (bin)')['Id'].count().reset_index()
    print("\nAge distribution:")
    print(age_counts)
    
    # Plot age distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=age_counts, 
               x='Age (bin)', 
               y='Id',
               color='blue')
    plt.title('Age Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_diversity(df):
    """Analyze and visualize employee diversity"""
    print("\n=== Employee Diversity Analysis ===")
    
    # Get diversity counts
    diversity = df.groupby(['Race', 'Gender'])['Id'].count().reset_index()
    
    # Plot diversity
    plt.figure(figsize=(16, 8))
    sns.barplot(data=diversity, 
               x='Race', 
               y='Id', 
               hue='Gender',
               palette='husl')
    plt.title('Employee Diversity by Race and Gender')
    plt.ylabel('Number of Employees')
    plt.tight_layout()
    plt.show()

def analyze_age_statistics(df):
    """Analyze age statistics by department"""
    print("\n=== Age Statistics Analysis ===")
    
    # Get average age by department
    avg_age = df.groupby('Department')['Age'].mean().reset_index()
    
    print("\nDepartments with highest and lowest average age:")
    print("Highest average age:")
    print(avg_age.sort_values('Age', ascending=False).head(1))
    print("\nLowest average age:")
    print(avg_age.sort_values('Age', ascending=True).head(1))
    
    # Plot average age by department
    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_age.sort_values('Age'), 
               x='Department', 
               y='Age',
               palette='mako')
    plt.title('Average Age by Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_tenure(df):
    """Analyze employee tenure"""
    print("\n=== Tenure Analysis ===")
    
    # Get tenure distribution
    tenure_counts = df.groupby('Tenure')['Id'].count().reset_index()
    
    # Plot tenure distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=tenure_counts.sort_values('Id'), 
               x='Tenure', 
               y='Id',
               color='blue')
    plt.title('Employee Tenure Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_hiring_trends(df):
    """Analyze hiring trends over time"""
    print("\n=== Hiring Trends Analysis ===")
    
    # Create hiring trends plot
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df.groupby(['Hire Month', 'Hire Year'])['Id'].count().reset_index(),
                x='Hire Month',
                y='Id',
                style='Hire Year',
                hue='Hire Year',
                marker='o',
                palette='tab10')
    plt.title('Hiring Trends Over Time')
    plt.xlabel('Month')
    plt.ylabel('Number of Hires')
    plt.tight_layout()
    plt.show()

def analyze_termination(df):
    """Analyze employee termination"""
    print("\n=== Termination Analysis ===")
    
    # Calculate termination rate
    termination_rate = df['Termdate'].dropna().count() / len(df)
    print(f"\nTermination rate: {termination_rate:.2%}")
    
    # Get termination distribution
    term_dist = df.groupby('Termdate (group)')['Id'].count().reset_index()
    
    # Plot termination distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=term_dist, 
               x='Termdate (group)', 
               y='Id',
               color='red')
    plt.title('Termination Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run all analyses"""
    print("Starting HR Data Analysis...")
    
    # Load and preprocess data
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        
        # Run all analyses
        analyze_department_distribution(df)
        analyze_age_distribution(df)
        analyze_diversity(df)
        analyze_age_statistics(df)
        analyze_tenure(df)
        analyze_hiring_trends(df)
        analyze_termination(df)
        
        # Run Prophet forecasting
        print("\n=== Hiring Forecast ===")
        try:
            # Prepare data for Prophet
            data = df.groupby('Hire Date')['Id'].count().reset_index()
            data = data.rename(columns={'Hire Date': 'ds', 'Id': 'y'})
            
            # Create and fit Prophet model
            m = Prophet()
            m.fit(data)
            
            # Make future predictions
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)
            
            # Plot forecast
            fig = plot_plotly(m, forecast)
            fig.show()
            
            # Plot components
            fig2 = m.plot_components(forecast)
            plt.show()
            
        except Exception as e:
            print(f"Error in Prophet analysis: {e}")

if __name__ == "__main__":
    main()

