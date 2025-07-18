# -*- coding: utf-8 -*-
"""
Titanic Dataset Data Manipulation

This script demonstrates various data manipulation techniques using the Titanic dataset.
It covers data loading, cleaning, transformation, and analysis operations.
"""

# Import required libraries for data analysis and visualization
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization

# Load the Titanic dataset from seaborn
# This dataset contains information about Titanic passengers
# including survival status, age, class, and other attributes
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
df

# Display all column names in the dataset
print("Column names in the dataset:")
df.columns

# Display dataset information including column data types and non-null counts
print("\nDataset information:")
df.info()

# Display the 'class' column
print("\nPassenger class values:")
df['class']

# Select and display specific columns from the dataset
print("\nSelected columns from the dataset:")
df[['sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alive']]

# Remove unnecessary columns from the dataset
# This helps in focusing on relevant features and reducing memory usage
print("\nDropping unnecessary columns...")
df = df.drop(columns=['survived','pclass','embarked','who','adult_male','alone'])

# Display the modified dataset
df

# Rename columns for better clarity and consistency
# 'class' is renamed to 'pclass' (passenger class)
# 'alive' is renamed to 'survived' for better clarity
print("\nRenaming columns...")
df.rename(columns={
    'class': 'pclass',
    'alive': 'survived'
}, inplace=True)

# Display the dataset with renamed columns
df

# Example of boolean evaluation in Python
# This shows how to compare values and create boolean masks
print("\nBoolean evaluation examples:")
x = 5
print(f"Is {x} equal to 0? {x == 0}")

# Create a boolean mask for passengers with no siblings/spouses
print("\nPassengers with no siblings/spouses:")
df['sibsp'] == 0

# Boolean operations in Python
print("\nBoolean operations:")
print("AND Operations:")
print(f"True & False = {True & False}")
print(f"True & True = {True & True}")
print(f"False & False = {False & False}")

print("\nOR Operations:")
print(f"True | False = {True | False}")
print(f"True | True = {True | True}")
print(f"False | False = {False | False}")

# Create boolean masks for passengers with no siblings/spouses and no parents/children
print("\nPassengers with no siblings/spouses and no parents/children:")
no_sibsp = (df['sibsp'] == 0)
no_parch = (df['parch'] == 0)
no_sibsp & no_parch

# Adding new columns to the DataFrame
print("\nAdding new columns to the dataset...")

# Create a new column 'alone' that is True when a passenger has no siblings/spouses or parents/children
df['alone'] = (df['sibsp'] == 0) & (df['parch'] == 0)
print("\nAfter adding 'alone' column:")
df

# Example of adding a column with sequential numbers
# Note: This will only work if the length matches the number of rows
# df['sample'] = range(200)  # Commented out as it would cause an error

# Add a column with a constant value
df['sample'] = 7
print("\nAfter adding 'sample' column with constant value 7:")
df

# Example of adding a column with a list
# Note: This will only work if the length of the list matches the number of rows
# df['sample2'] = [1,2,3,4]  # Commented out as it would cause an error

# Modifying an existing column
print("\nModifying the 'sample' column to have a new value...")
df['sample'] = 8
print("After modification:")
df

# Mapping categorical values to numerical values
print("\nMapping 'survived' column from text to numerical values...")
survival_mapping = {
    'no': 0,   # Did not survive
    'yes': 1    # Survived
}

# Create a new column with numerical values
df['survived_num'] = df['survived'].replace(survival_mapping)
print("After mapping 'survived' to numerical values:")
df

# Analyzing unique values in the dataset
print("\nAnalyzing unique values in the dataset:")

# Get unique deck values (cabin decks where passengers stayed)
print("\nUnique deck values:")
df['deck'].unique()

# Count unique embarkation towns
print(f"\nNumber of unique embarkation towns: {df['embark_town'].nunique()}")

# Analyzing value distribution
print("\nDistribution of passengers by embarkation town:")
print(df['embark_town'].value_counts())

# Creating age groups (buckets) for analysis
print("\nCreating age groups by decade:")

# Calculate age group (e.g., 23 becomes 20, 35 becomes 30, etc.)
age_groups = df['age'] // 10 * 10
print("\nAge groups (decades):")
print(age_groups.head())

# Add age group as a new column to the dataframe
df['age_bucket'] = df['age'] // 10 * 10
print("\nAfter adding 'age_bucket' column:")
df[['age', 'age_bucket']].head()

# Grouping and aggregating data
print("\n--- Grouping and Aggregating Data ---")

# Calculate average age by gender
print("\nAverage age by gender:")
print(df.groupby('sex').age.mean())

# Calculate mean of all numerical columns grouped by gender
print("\nMean of all numerical columns by gender:")
print(df.groupby('sex').age.mean())

# Alternative way to calculate average age by gender
print("\nAlternative way to calculate average age by gender:")
male_avg_age = df[df.sex == 'male'].age.mean()
female_avg_age = df[df.sex == 'female'].age.mean()
print(f"Male average age: {male_avg_age:.1f}, Female average age: {female_avg_age:.1f}")

# Multiple aggregation functions on age by gender
print("\nAge statistics by gender (min, mean, max):")
print(df.groupby('sex').age.agg([np.min, np.mean, np.max]))

# Calculate survival rate by age group
print("\nSurvival rate by age group:")
print(df.groupby('age_bucket').survived_num.mean())

# Calculate survival rate by both gender and age group with counts
print("\nSurvival rate by gender and age group with group sizes:")
print(df.groupby(['sex', 'age_bucket']).survived_num.agg([np.mean, np.size]))

# Sorting data
print("\n--- Sorting Data ---")
print("\nSorting passengers by age (ascending):")
sorted_by_age = df.sort_values(by='age')
sorted_by_age[['age', 'sex', 'survived']].head()

# Handling missing values
print("\n--- Handling Missing Values ---")
print("\nCurrent dataset info (showing missing values):")
df.info()

# Option 1: Drop rows with any missing values
print("\nDropping rows with any missing values:")
print("Rows before:", len(df))
print("Rows after:", len(df.dropna()))

# Option 2: Drop rows with missing values in specific columns
print("\nDropping rows with missing 'embark_town' values:")
print("Rows before:", len(df))
print("Rows after:", len(df.dropna(subset=['embark_town'])))

# Option 3: Fill missing age values with 0
print("\nFilling missing age values with 0:")
print(df['age'].fillna(0).head())

# Option 4: Fill missing age values with the mean age
print("\nFilling missing age values with mean age:")
mean_age = df['age'].mean()
print(f"Mean age: {mean_age:.1f}")
print(df['age'].fillna(mean_age).head())

# Applying functions to data
print("\n--- Applying Functions to Data ---")

# Basic arithmetic operation on a column
print("\nAdding 5 to each fare value:")
print((df['fare'] + 5).head())

# More complex mathematical operation
# Formula: x³*3 + x² + 7x + 5
print("\nApplying polynomial function to fare values:")
fare_polynomial = df['fare']**3 * 3 + df['fare']**2 + 7*df['fare'] + 5
print(fare_polynomial.head())

# Define a custom function for the same calculation
def my_function(x):
    """
    Calculate a polynomial function: 3x³ + x² + 7x + 5
    
    Args:
        x: Input value
    Returns:
        Result of the polynomial calculation
    """
    return 3*x**3 + x**2 + 7*x + 5

# Test the function with a single value
print("\nTesting the function with x=5:")
print(f"Result: {my_function(5)}")

# Apply the function to the 'fare' column
print("\nApplying the function to 'fare' column:")
print(df['fare'].apply(my_function).head())

# The same operation using a lambda function
print("\nSame operation using lambda function:")
print(df['fare'].apply(lambda x: 3*x**3 + x**2 + 7*x + 5).head())

# Apply function with adjustment
print("\nApply function and subtract 5 from result:")
print(df['fare'].apply(lambda x: my_function(x) - 5).head())

# Apply function to multiple columns
print("\nApply function to multiple columns (age and fare):")
print(df[['age', 'fare']].apply(my_function).head())

# String operations on text data
print("\n--- String Operations ---")

# Ensure we've dropped rows with missing embark_town
df = df.dropna(subset=['embark_town'])

# Extract first letter of embark_town
print("\nFirst letter of each embarkation town:")
print(df['embark_town'].map(lambda x: x[0]).head())

# Convert town names to lowercase
print("\nEmbarkation towns in lowercase:")
print(df['embark_town'].map(lambda x: x.lower()).head())

# Selecting columns by data type
print("\n--- Selecting Columns by Data Type ---")

# Select only numeric columns (float and int)
print("\nNumeric columns in the dataset:")
numeric_cols = df.select_dtypes(include=['float', 'int'])
print(numeric_cols.head())

# Apply function to numeric columns only
print("\nApply polynomial function to numeric columns:")
print(numeric_cols.apply(my_function).head())

# Reordering columns
print("\n--- Reordering Columns ---")

# Get current column order
print("\nCurrent column order:")
print(df.columns.tolist())

# Create a new DataFrame with reordered columns
print("\nSelecting specific columns in a new order:")
reordered_df = df[['age', 'survived', 'alone', 'age_bucket']]
print(reordered_df.head())

# Combining values from different columns
print("\n--- Combining Column Values ---")

# Function to create a descriptive string from multiple columns
def print_statement(x):
    """
    Create a formatted string with fare and class information for a passenger.
    
    Args:
        x: A pandas Series representing a row of the DataFrame
    Returns:
        A formatted string with fare and class information
    """
    return f"Passenger fare is: {x['fare']:.2f} and class is {x['pclass']}"

# Apply the function to each row of the DataFrame
print("\nGenerating passenger fare and class statements:")
print(df.apply(print_statement, axis=1).head())

# Function to perform a calculation using multiple columns
def multi_column_function(x):
    """
    Calculate fare plus 5 (example of a simple transformation).
    
    Args:
        x: A pandas Series representing a row of the DataFrame
    Returns:
        The fare value plus 5
    """
    return x['fare'] + 5

# Apply the function to each row
print("\nCalculating fare + 5 for each passenger:")
print(df.apply(multi_column_function, axis=1).head())

# Select specific columns
print("\nDisplaying fare and pclass columns:")
print(df[['fare', 'pclass']].head())

# Create a new column by applying a function to an existing column
print("\nCreating a new column 'fare_plus_5' by adding 5 to each fare value:")
df['fare_plus_5'] = df[['fare']].apply(lambda x: x + 5)

# Display the original and new fare values
print("\nOriginal fare vs fare + 5:")
print(df[['fare', 'fare_plus_5']].head())
