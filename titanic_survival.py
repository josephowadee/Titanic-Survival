# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic.head())

# Check for missing values
print("\nMissing values in each column:")
print(titanic.isnull().sum())

# Handle missing values
# Fill missing 'age' values with the median age
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Drop rows with missing 'embarked' values
titanic.dropna(subset=['embarked'], inplace=True)

# Verify that there are no missing values left
print("\nMissing values after handling:")
print(titanic.isnull().sum())

# Basic statistical analysis
print("\nBasic statistical summary of the dataset:")
print(titanic.describe(include='all'))

# Data visualization

## Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(titanic['age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

## Survival Rate by Class
plt.figure(figsize=(10, 6))
sns.countplot(x='class', hue='survived', data=titanic, palette='pastel')
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

## Age vs Fare Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='fare', hue='survived', data=titanic, palette='coolwarm', alpha=0.7)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
