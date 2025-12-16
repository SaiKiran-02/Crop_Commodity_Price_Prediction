import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path_or_df):
    """Load and preprocess data"""
    try:
        df = pd.read_csv(file_path_or_df)
        if 'Arrival_Date' in df.columns:
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d-%m-%Y')
        return df
    except:
        return pd.DataFrame()


def safe_cat_analysis(df, col, top=15, figsize=(20, 10)):
    if df.empty or col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=24, color='darkgreen')
        ax.set_title(f'No Data for {col}', fontsize=28, fontweight='bold', color='darkgreen')
        ax.axis('off')
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    value_counts = df[col].value_counts()
    bars = value_counts[:top].plot.bar(ax=ax, color='#2E8B57', edgecolor='darkgreen', linewidth=3, alpha=0.85)
    ax.set_title(f'🌾 Top {min(top, len(value_counts))} {col}', fontsize=28, fontweight='bold', pad=20)
    ax.set_xlabel(col, fontsize=20, fontweight='bold')
    ax.set_ylabel('Count', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                f'{int(height):,}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    return fig


def safe_price_dist(df, col, figsize=(14, 10)):
    if df.empty or col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=24, color='darkgreen')
        ax.set_title(f'No Data for {col}', fontsize=28, fontweight='bold', color='darkgreen')
        ax.axis('off')
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df[col], bins=30, kde=True, ax=ax, color='#228B22', alpha=0.75, edgecolor='darkgreen', linewidth=2)
    ax.set_title(f'🌱 {col} Distribution', fontsize=28, fontweight='bold', pad=20)
    ax.set_xlabel(col, fontsize=20, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_time_series_plot(df, title="Average Modal Price Over Time"):
    if df.empty or 'Arrival_Date' not in df.columns or 'Modal Price' not in df.columns:
        return None
    time_data = df.groupby('Arrival_Date')['Modal Price'].mean().reset_index()
    fig = px.line(time_data, x='Arrival_Date', y='Modal Price', title=f'🌾 {title}', markers=True,
                  color_discrete_sequence=['#228B22'])
    fig.update_layout(font_size=14, title_font_size=24)
    return fig


def create_correlation_heatmap(df):
    price_cols = ['Min Price', 'Max Price', 'Modal Price']
    if len(df) > 1 and all(col in df.columns for col in price_cols):
        fig, ax = plt.subplots(figsize=(14, 10))
        corr_matrix = df[price_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='YlGn', center=0, ax=ax, annot_kws={'size': 20}, fmt='.2f')
        ax.set_title('🔗 Price Correlation Heatmap', fontsize=28, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    return None
