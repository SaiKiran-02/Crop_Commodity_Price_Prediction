import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')


# ALL EDA FUNCTIONS DIRECTLY HERE - NO IMPORTS!
def load_data(file_path_or_df):
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
    ax.set_title(f'Top {min(top, len(value_counts))} {col}', fontsize=28, fontweight='bold', pad=20)
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


def create_box_plot(df, col, price_col='Modal Price', top_n=6):
    if df.empty or col not in df.columns or price_col not in df.columns:
        return None
    top_categories = df[col].value_counts().head(top_n).index
    plot_data = df[df[col].isin(top_categories)]
    fig = px.box(plot_data, x=col, y=price_col, title=f"🌾 {price_col} by Top {col}",
                 color_discrete_sequence=['#228B22'])
    fig.update_layout(xaxis_tickangle=45, font_size=14, title_font_size=22)
    return fig


def get_dataset_stats(df):
    if df.empty:
        return {'records': 0, 'commodities': 0, 'states': 0, 'avg_price': 0}
    return {
        'records': len(df),
        'commodities': df['Commodity'].nunique() if 'Commodity' in df.columns else 0,
        'states': df['State'].nunique() if 'State' in df.columns else 0,
        'avg_price': df['Modal Price'].mean() if 'Modal Price' in df.columns else 0
    }


# STREAMLIT APP STARTS HERE
st.set_page_config(page_title="Agriculture Price Dashboard", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

st.title("🌾 Agriculture Commodities Price Dashboard")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])

if uploaded_file is None:
    st.info("👆 Please upload your CSV file to get started!")
    st.stop()

df = load_data(uploaded_file)
if df.empty:
    st.error("❌ Failed to load data. Please check your CSV file format.")
    st.stop()

st.success(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Sidebar
st.sidebar.header("🔍 Filters")
all_commodities = df['Commodity'].unique()
all_states = df['State'].unique()
top_commodities = df['Commodity'].value_counts().head(5).index.tolist()
top_states = df['State'].value_counts().head(5).index.tolist()

commodities = st.sidebar.multiselect("Commodities", all_commodities, default=top_commodities)
states = st.sidebar.multiselect("States", all_states, default=top_states)

if not commodities: commodities = all_commodities
if not states: states = all_states

filtered_df = df[(df['Commodity'].isin(commodities)) & (df['State'].isin(states))]
if filtered_df.empty:
    st.warning("⚠️ No data matches filters. Showing full dataset.")
    filtered_df = df

# Metrics
col1, col2, col3, col4 = st.columns(4)
stats = get_dataset_stats(filtered_df)
col1.metric("Records", stats['records'])
col2.metric("Commodities", stats['commodities'])
col3.metric("States", stats['states'])
col4.metric("Avg Price", f"₹{stats['avg_price']:.0f}" if stats['avg_price'] > 0 else "N/A")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Categorical", "Distributions", "Time Series", "Advanced"])

with tab1:
    st.subheader("Dataset Overview")

    # Option 1: Show ALL data (scrollable)
    if st.checkbox("👁️ Show FULL Dataset", value=False):
        st.dataframe(filtered_df, use_container_width=True, height=600)
    else:
        st.dataframe(filtered_df.head(10), use_container_width=True)

    # Option 2: Dataset Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(filtered_df))
    with col2:
        st.metric("Columns", len(filtered_df.columns))
    with col3:
        st.metric("🕒 Date Range",
                  f"{filtered_df['Arrival_Date'].min().date()} to {filtered_df['Arrival_Date'].max().date()}")
    with col4:
        st.metric("Price Range", f"₹{filtered_df['Modal Price'].min():.0f} - ₹{filtered_df['Modal Price'].max():.0f}")

    # Option 3: Column Info
    st.markdown("Column Information")
    st.json({col: str(filtered_df[col].dtype) for col in filtered_df.columns})

    # Option 4: Download Full Data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="💾 Download Full Dataset as CSV",
        data=csv,
        file_name='agriculture_commodities_filtered.csv',
        mime='text/csv'
    )

    # Option 5: Sample + Full Preview Toggle
    st.markdown("Quick Preview (First 20 rows)")
    st.dataframe(filtered_df.head(20), use_container_width=True)

with tab2:
    st.markdown("Top Commodities")
    st.pyplot(safe_cat_analysis(filtered_df, 'Commodity', 15))
    st.markdown("Top States")
    st.pyplot(safe_cat_analysis(filtered_df, 'State', 10))

with tab3:
    st.markdown("Price Distributions")
    st.pyplot(safe_price_dist(filtered_df, 'Modal Price'))
    st.pyplot(safe_price_dist(filtered_df, 'Max Price'))
    st.pyplot(safe_price_dist(filtered_df, 'Min Price'))

with tab4:
    st.subheader("Main Time Series")
    timeseries_fig = create_time_series_plot(filtered_df)
    if timeseries_fig:
        st.plotly_chart(timeseries_fig, use_container_width=True)

    st.subheader("Date Range Filter")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", filtered_df['Arrival_Date'].min().date())
    end_date = col2.date_input("End Date", filtered_df['Arrival_Date'].max().date())

    time_filtered = filtered_df[(filtered_df['Arrival_Date'] >= pd.to_datetime(start_date)) &
                                (filtered_df['Arrival_Date'] <= pd.to_datetime(end_date))]
    if not time_filtered.empty:
        filtered_timeseries = create_time_series_plot(time_filtered, "Filtered Date Range")
        if filtered_timeseries:
            st.plotly_chart(filtered_timeseries, use_container_width=True)

with tab5:
    st.subheader("🔗 Price Correlation")
    corr_fig = create_correlation_heatmap(filtered_df)
    if corr_fig:
        st.pyplot(corr_fig)

    # st.subheader("Price by Top States")
    state_box = create_box_plot(filtered_df, 'State')
    if state_box:
        st.plotly_chart(state_box, use_container_width=True)

    # st.subheader("Price by Top Commodities")
    commodity_box = create_box_plot(filtered_df, 'Commodity')
    if commodity_box:
        st.plotly_chart(commodity_box, use_container_width=True)

st.markdown("---")
