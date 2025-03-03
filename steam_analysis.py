#!/usr/bin/env python3
"""
Steam Games Dataset Analysis

This script analyzes the Steam Games dataset from Kaggle:
https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data?select=games_may2024_full.csv

To run this script:
1. Download the dataset from Kaggle
2. Place it in the 'data' directory
3. Run this script

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set plot styles
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

def load_data(file_path):
    """Load the dataset from the specified path."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of games: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    return df

def explore_data(df):
    """Perform basic exploratory data analysis."""
    print("\n--- Basic Information ---")
    print(df.info())
    
    print("\n--- Missing Values ---")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False))
    
    print("\n--- Basic Statistics ---")
    print(df.describe())
    
    return df

def analyze_price(df):
    """Analyze price distribution and statistics."""
    if 'price' not in df.columns:
        print("Price column not found in dataset.")
        return
    
    print("\n--- Price Analysis ---")
    
    # Price statistics
    print(f"Average price: ${df['price'].mean():.2f}")
    print(f"Median price: ${df['price'].median():.2f}")
    print(f"Maximum price: ${df['price'].max():.2f}")
    print(f"Percentage of free games: {(df['price'] == 0).mean() * 100:.2f}%")
    
    # Price distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Distribution of Game Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.xlim(0, 100)  # Limiting x-axis to focus on the main distribution
    plt.savefig('price_distribution.png')
    plt.close()
    print("Price distribution plot saved as 'price_distribution.png'")

def analyze_ratings(df):
    """Analyze ratings distribution and statistics."""
    if 'rating' not in df.columns:
        print("Rating column not found in dataset.")
        return
    
    print("\n--- Ratings Analysis ---")
    
    # Convert ratings to numeric if needed
    if df['rating'].dtype == 'object':
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['rating_numeric'] = df['rating']
    
    # Rating statistics
    print(f"Average rating: {df['rating_numeric'].mean():.2f}")
    print(f"Median rating: {df['rating_numeric'].median():.2f}")
    
    # Rating distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(df['rating_numeric'].dropna(), bins=20, kde=True)
    plt.title('Distribution of Game Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('rating_distribution.png')
    plt.close()
    print("Rating distribution plot saved as 'rating_distribution.png'")
    
    return df

def analyze_release_dates(df):
    """Analyze release date trends."""
    if 'release_date' not in df.columns:
        print("Release date column not found in dataset.")
        return df
    
    print("\n--- Release Date Analysis ---")
    
    # Convert release date to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Extract year
    df['release_year'] = df['release_date'].dt.year
    
    # Games released by year
    plt.figure(figsize=(15, 6))
    year_counts = df['release_year'].value_counts().sort_index()
    year_counts.plot(kind='bar')
    plt.title('Number of Games Released by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=45)
    plt.savefig('games_by_year.png')
    plt.close()
    print("Games by year plot saved as 'games_by_year.png'")
    
    return df

def analyze_genres(df):
    """Analyze game genres."""
    if 'genres' not in df.columns:
        print("Genres column not found in dataset.")
        return
    
    print("\n--- Genre Analysis ---")
    
    # Split genres into a list if they're in a string format
    if df['genres'].dtype == 'object':
        # Check if genres are stored as strings that need to be split
        sample_genre = df['genres'].dropna().iloc[0] if not df['genres'].dropna().empty else ''
        
        if isinstance(sample_genre, str) and (',' in sample_genre or ';' in sample_genre):
            # Determine the separator
            separator = ',' if ',' in sample_genre else ';'
            
            # Extract all genres
            all_genres = []
            for genres_str in df['genres'].dropna():
                genres_list = [genre.strip() for genre in genres_str.split(separator)]
                all_genres.extend(genres_list)
                
            # Count genre occurrences
            genre_counts = pd.Series(all_genres).value_counts()
            
            # Plot top genres
            plt.figure(figsize=(12, 8))
            genre_counts.head(20).plot(kind='barh')
            plt.title('Top 20 Game Genres')
            plt.xlabel('Number of Games')
            plt.ylabel('Genre')
            plt.savefig('top_genres.png')
            plt.close()
            print("Top genres plot saved as 'top_genres.png'")
        else:
            # If genres are not separated by commas or semicolons
            top_genres = df['genres'].value_counts().head(20)
            plt.figure(figsize=(12, 8))
            top_genres.plot(kind='barh')
            plt.title('Top 20 Game Genres')
            plt.xlabel('Number of Games')
            plt.ylabel('Genre')
            plt.savefig('top_genres.png')
            plt.close()
            print("Top genres plot saved as 'top_genres.png'")

def analyze_developers(df):
    """Analyze game developers."""
    if 'developer' not in df.columns:
        print("Developer column not found in dataset.")
        return
    
    print("\n--- Developer Analysis ---")
    
    # Top developers by number of games
    top_developers = df['developer'].value_counts().head(20)
    
    plt.figure(figsize=(12, 8))
    top_developers.plot(kind='barh')
    plt.title('Top 20 Developers by Number of Games')
    plt.xlabel('Number of Games')
    plt.ylabel('Developer')
    plt.savefig('top_developers.png')
    plt.close()
    print("Top developers plot saved as 'top_developers.png'")

def analyze_correlations(df):
    """Analyze correlations between numerical features."""
    print("\n--- Correlation Analysis ---")
    
    # Select numerical columns for correlation analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('correlation_matrix.png')
    plt.close()
    print("Correlation matrix plot saved as 'correlation_matrix.png'")

def analyze_price_vs_rating(df):
    """Analyze relationship between price and rating."""
    if 'price' not in df.columns or 'rating_numeric' not in df.columns:
        print("Price or rating columns not found in dataset.")
        return
    
    print("\n--- Price vs. Rating Analysis ---")
    
    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='price', y='rating_numeric', data=df, alpha=0.5)
    plt.title('Game Price vs. Rating')
    plt.xlabel('Price ($)')
    plt.ylabel('Rating')
    plt.xlim(0, 100)  # Limit x-axis to focus on the main distribution
    plt.savefig('price_vs_rating.png')
    plt.close()
    print("Price vs. rating plot saved as 'price_vs_rating.png'")
    
    # Calculate correlation
    correlation = df['price'].corr(df['rating_numeric'])
    print(f"Correlation between price and rating: {correlation:.3f}")

def main():
    """Main function to run the analysis."""
    # Define the file path
    file_path = 'data/games_may2024_full.csv'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please download the dataset from Kaggle and place it in the 'data' directory.")
        return
    
    # Load the data
    df = load_data(file_path)
    
    # Explore the data
    df = explore_data(df)
    
    # Perform various analyses
    analyze_price(df)
    df = analyze_ratings(df)
    df = analyze_release_dates(df)
    analyze_genres(df)
    analyze_developers(df)
    analyze_correlations(df)
    analyze_price_vs_rating(df)
    
    print("\nAnalysis complete! Check the generated plots for visualizations.")

if __name__ == "__main__":
    main()