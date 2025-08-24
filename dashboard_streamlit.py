# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="🚲 Dashboard Analisis Bike Sharing",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
         color: black;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
             color: black;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        day_df = pd.read_csv('day.csv')
        hour_df = pd.read_csv('hour.csv')
    except FileNotFoundError:
        st.error("⚠️ File data tidak ditemukan. Pastikan file 'day.csv' dan 'hour.csv' tersedia.")
        return None, None

    # Data preprocessing
    for df in [day_df, hour_df]:
        df['dteday'] = pd.to_datetime(df['dteday'])
        df['year'] = df['dteday'].dt.year
        df['month'] = df['dteday'].dt.month
        df['day'] = df['dteday'].dt.day
        df['day_of_week'] = df['dteday'].dt.dayofweek

    # Mapping dictionaries
    season_dict = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weathersit_dict = {1: 'Clear', 2: 'Misty', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    weekday_dict = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    # Apply mappings
    for df in [day_df, hour_df]:
        df['season_name'] = df['season'].map(season_dict)
        df['weathersit_name'] = df['weathersit'].map(weathersit_dict)
        df['month_name'] = df['month'].map(month_dict)
        df['weekday_name'] = df['day_of_week'].map(weekday_dict)
    
    hour_df['is_weekend'] = hour_df['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    return day_df, hour_df

def create_rfm_analysis(day_df):
    """
    Perform RFM-like analysis for bike sharing data
    Adapted for bike sharing context:
    - Recency: Days since last high usage
    - Frequency: Number of high-usage days
    - Monetary: Average daily rentals
    """
    
    # Define high usage threshold (top 25% of daily rentals)
    high_usage_threshold = day_df['cnt'].quantile(0.75)
    
    # Calculate RFM metrics for each month
    monthly_data = []
    
    for month in day_df['month'].unique():
        month_data = day_df[day_df['month'] == month]
        
        # Recency: Days since last high usage day in the month
        high_usage_days = month_data[month_data['cnt'] >= high_usage_threshold]
        if len(high_usage_days) > 0:
            last_high_usage = high_usage_days['dteday'].max()
            month_end = month_data['dteday'].max()
            recency = (month_end - last_high_usage).days
        else:
            recency = 30  # Max recency if no high usage days
        
        # Frequency: Number of high usage days in the month
        frequency = len(high_usage_days)
        
        # Monetary: Average daily rentals in the month
        monetary = month_data['cnt'].mean()
        
        monthly_data.append({
            'month': month,
            'month_name': month_data['month_name'].iloc[0],
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        })
    
    rfm_df = pd.DataFrame(monthly_data)
    
    # Create RFM scores (1-5 scale)
    rfm_df['R_score'] = pd.cut(rfm_df['recency'], bins=5, labels=[5,4,3,2,1])
    rfm_df['F_score'] = pd.cut(rfm_df['frequency'], bins=5, labels=[1,2,3,4,5])
    rfm_df['M_score'] = pd.cut(rfm_df['monetary'], bins=5, labels=[1,2,3,4,5])
    
    # Convert to numeric
    rfm_df['R_score'] = rfm_df['R_score'].astype(int)
    rfm_df['F_score'] = rfm_df['F_score'].astype(int)
    rfm_df['M_score'] = rfm_df['M_score'].astype(int)
    
    # Create RFM combined score
    rfm_df['RFM_score'] = rfm_df['R_score'] + rfm_df['F_score'] + rfm_df['M_score']
    
    # Segment customers
    def rfm_segment(row):
        if row['RFM_score'] >= 12:
            return 'High Performance'
        elif row['RFM_score'] >= 9:
            return 'Medium Performance'
        else:
            return 'Low Performance'
    
    rfm_df['segment'] = rfm_df.apply(rfm_segment, axis=1)
    
    return rfm_df

def create_clustering_analysis(day_df):
    """
    Perform manual clustering based on usage patterns
    """
    # Create features for clustering
    day_df['temp_range'] = pd.cut(day_df['temp'], bins=3, labels=['Cold', 'Moderate', 'Warm'])
    day_df['humidity_range'] = pd.cut(day_df['hum'], bins=3, labels=['Low', 'Medium', 'High'])
    day_df['usage_level'] = pd.cut(day_df['cnt'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Manual clustering based on weather and usage patterns
    def weather_usage_cluster(row):
        if row['weathersit'] == 1 and row['usage_level'] == 'High':
            return 'Optimal Conditions'
        elif row['weathersit'] <= 2 and row['usage_level'] == 'Medium':
            return 'Good Conditions'
        elif row['weathersit'] >= 3 or row['usage_level'] == 'Low':
            return 'Poor Conditions'
        else:
            return 'Average Conditions'
    
    day_df['weather_usage_cluster'] = day_df.apply(weather_usage_cluster, axis=1)
    
    return day_df

# Main Dashboard
def main():
    # Load data
    day_df, hour_df = load_data()
    
    if day_df is None or hour_df is None:
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚲 Bike Sharing Analysis Dashboard</h1>
        <p>Comprehensive analysis of bike sharing patterns and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Year filter
        years = sorted(day_df['year'].unique())
        selected_year = st.selectbox("📅 Select Year", years, index=len(years)-1)
        
        # Analysis type
        analysis_type = st.radio(
            "📊 Analysis Focus",
            ["Overview", "Seasonal & Weather", "Temporal Patterns", "Advanced Analytics"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Business Questions")
        st.markdown("""
        1. **How do seasonal changes and weather conditions affect bike sharing demand?**
        2. **What are the daily and hourly usage patterns, and how do they differ between weekdays and weekends?**
        """)
    
    # Filter data
    day_filtered = day_df[day_df['year'] == selected_year]
    hour_filtered = hour_df[hour_df['year'] == selected_year]
    
    # Overview Tab
    if analysis_type == "Overview":
        st.header("📈 Key Performance Indicators")
        
        # Calculate metrics
        total_rides = day_filtered['cnt'].sum()
        avg_daily_rides = day_filtered['cnt'].mean()
        casual_percentage = (day_filtered['casual'].sum() / total_rides) * 100
        registered_percentage = (day_filtered['registered'].sum() / total_rides) * 100
        peak_day = day_filtered.loc[day_filtered['cnt'].idxmax(), 'dteday'].strftime('%Y-%m-%d')
        peak_rides = day_filtered['cnt'].max()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Rides", f"{total_rides:,}", delta=None)
        with col2:
            st.metric("Daily Average", f"{avg_daily_rides:.0f}", delta=None)
        with col3:
            st.metric("Casual Users", f"{casual_percentage:.1f}%", delta=None)
        with col4:
            st.metric("Registered Users", f"{registered_percentage:.1f}%", delta=None)
        with col5:
            st.metric("Peak Day Rides", f"{peak_rides:,}", delta=f"{peak_day}")
        
        # Time series plot
        st.subheader("📊 Daily Bike Sharing Trends")
        
        fig = px.line(day_filtered, x='dteday', y='cnt', 
                     title=f'Daily Bike Rentals Trend - {selected_year}',
                     labels={'dteday': 'Date', 'cnt': 'Number of Rentals'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # User type breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            user_data = pd.DataFrame({
                'User Type': ['Casual', 'Registered'],
                'Count': [day_filtered['casual'].sum(), day_filtered['registered'].sum()]
            })
            
            fig = px.pie(user_data, values='Count', names='User Type', 
                        title='User Type Distribution',
                        color_discrete_sequence=['#ff9999', '#66b3ff'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_trend = day_filtered.groupby('month_name')[['casual', 'registered']].sum().reset_index()
            
            fig = px.bar(monthly_trend, x='month_name', y=['casual', 'registered'],
                        title='Monthly User Distribution',
                        labels={'month_name': 'Month', 'value': 'Number of Rentals'},
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal & Weather Analysis
    elif analysis_type == "Seasonal & Weather":
        st.header("🌤️ Seasonal and Weather Impact Analysis")
        
        # Business Question 1 Analysis
        st.markdown("""
        <div class="insight-box">
        <h4>📋 Business Question 1: How do seasonal changes and weather conditions affect bike sharing demand?</h4>
        <p>This analysis examines the relationship between environmental factors and bike usage patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seasonal analysis
            seasonal_data = day_filtered.groupby('season_name')['cnt'].agg(['mean', 'sum', 'count']).reset_index()
            
            fig = px.bar(seasonal_data, x='season_name', y='mean',
                        title='Average Daily Rentals by Season',
                        labels={'season_name': 'Season', 'mean': 'Average Daily Rentals'},
                        color='mean',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal statistics
            st.markdown("**Seasonal Performance:**")
            for _, row in seasonal_data.iterrows():
                st.write(f"• **{row['season_name']}**: {row['mean']:.0f} avg daily rentals")
        
        with col2:
            # Weather analysis
            weather_data = day_filtered.groupby('weathersit_name')['cnt'].agg(['mean', 'sum', 'count']).reset_index()
            
            fig = px.bar(weather_data, x='weathersit_name', y='mean',
                        title='Average Daily Rentals by Weather Condition',
                        labels={'weathersit_name': 'Weather Condition', 'mean': 'Average Daily Rentals'},
                        color='mean',
                        color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather statistics
            st.markdown("**Weather Impact:**")
            for _, row in weather_data.iterrows():
                st.write(f"• **{row['weathersit_name']}**: {row['mean']:.0f} avg daily rentals")
        
        # Combined analysis
        st.subheader("🌈 Season-Weather Interaction Analysis")
        
        season_weather = day_filtered.groupby(['season_name', 'weathersit_name'])['cnt'].mean().reset_index()
        
        fig = px.sunburst(season_weather, 
                         path=['season_name', 'weathersit_name'], 
                         values='cnt',
                         title='Hierarchical View: Season → Weather → Average Rentals')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Environmental Factors Correlation")
        
        corr_data = day_filtered[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
        
        fig = px.imshow(corr_data, 
                       title='Correlation Matrix: Environmental Factors vs Bike Rentals',
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal Patterns Analysis
    elif analysis_type == "Temporal Patterns":
        st.header("⏰ Temporal Usage Patterns Analysis")
        
        # Business Question 2 Analysis
        st.markdown("""
        <div class="insight-box">
        <h4>📋 Business Question 2: What are the daily and hourly usage patterns, and how do they differ between weekdays and weekends?</h4>
        <p>This analysis reveals time-based patterns in bike sharing usage.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekly pattern
            weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_data = day_filtered.groupby('weekday_name')['cnt'].mean().reindex(weekday_order)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_data.index, 
                y=daily_data.values,
                mode='lines+markers',
                name='Average Rentals',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Weekly Usage Pattern',
                xaxis_title='Day of Week',
                yaxis_title='Average Daily Rentals',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weekend vs Weekday comparison
            weekend_comparison = hour_filtered.groupby(['hr', 'is_weekend'])['cnt'].mean().reset_index()
            
            fig = px.line(weekend_comparison, x='hr', y='cnt', color='is_weekend',
                         title='Hourly Usage: Weekday vs Weekend',
                         labels={'hr': 'Hour of Day', 'cnt': 'Average Rentals', 'is_weekend': 'Day Type'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Hourly heatmap
        st.subheader("🔥 Hourly Usage Heatmap")
        
        # Create hour-weekday heatmap data
        heatmap_data = hour_filtered.pivot_table(
            values='cnt', 
            index='weekday_name', 
            columns='hr', 
            aggfunc='mean'
        )
        
        # Reorder days
        heatmap_data = heatmap_data.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        fig = px.imshow(heatmap_data, 
                       title='Average Bike Rentals by Hour and Day of Week',
                       labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Avg Rentals'},
                       color_continuous_scale='viridis',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak hours analysis
        st.subheader("🎯 Peak Hours Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weekday_hours = hour_filtered[hour_filtered['is_weekend'] == 'Weekday'].groupby('hr')['cnt'].mean()
            peak_weekday_hour = weekday_hours.idxmax()
            peak_weekday_value = weekday_hours.max()
            st.metric("Weekday Peak Hour", f"{peak_weekday_hour}:00", f"{peak_weekday_value:.0f} rentals")
        
        with col2:
            weekend_hours = hour_filtered[hour_filtered['is_weekend'] == 'Weekend'].groupby('hr')['cnt'].mean()
            peak_weekend_hour = weekend_hours.idxmax()
            peak_weekend_value = weekend_hours.max()
            st.metric("Weekend Peak Hour", f"{peak_weekend_hour}:00", f"{peak_weekend_value:.0f} rentals")
        
        with col3:
            overall_hours = hour_filtered.groupby('hr')['cnt'].mean()
            peak_overall_hour = overall_hours.idxmax()
            peak_overall_value = overall_hours.max()
            st.metric("Overall Peak Hour", f"{peak_overall_hour}:00", f"{peak_overall_value:.0f} rentals")
    
    # Advanced Analytics
    elif analysis_type == "Advanced Analytics":
        st.header("🧠 Advanced Analytics")
        
        tab1, tab2, tab3 = st.tabs(["RFM Analysis", "Usage Clustering", "Performance Segmentation"])
        
        with tab1:
            st.subheader("📊 RFM-Style Analysis (Monthly Performance)")
            st.markdown("""
            Adapted RFM analysis for bike sharing data:
            - **Recency**: Days since last high-usage day in the month
            - **Frequency**: Number of high-usage days per month
            - **Monetary**: Average daily rentals per month
            """)
            
            rfm_df = create_rfm_analysis(day_filtered)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter_3d(rfm_df, x='recency', y='frequency', z='monetary',
                                   color='segment', size='RFM_score',
                                   title='3D RFM Analysis: Monthly Performance',
                                   labels={'recency': 'Recency (days)', 
                                          'frequency': 'Frequency (high-usage days)',
                                          'monetary': 'Monetary (avg rentals)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                segment_counts = rfm_df['segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                            title='Monthly Performance Segments')
                st.plotly_chart(fig, use_container_width=True)
            
            # RFM detailed table
            st.subheader("📋 Detailed RFM Analysis")
            st.dataframe(rfm_df[['month_name', 'recency', 'frequency', 'monetary', 'RFM_score', 'segment']])
        
        with tab2:
            st.subheader("🎯 Usage Pattern Clustering")
            
            clustered_data = create_clustering_analysis(day_filtered.copy())
            
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_summary = clustered_data.groupby('weather_usage_cluster').agg({
                    'cnt': ['count', 'mean'],
                    'temp': 'mean',
                    'hum': 'mean'
                }).round(2)
                
                cluster_counts = clustered_data['weather_usage_cluster'].value_counts()
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                            title='Distribution of Usage Clusters',
                            labels={'x': 'Cluster', 'y': 'Number of Days'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(clustered_data, x='temp', y='cnt', 
                               color='weather_usage_cluster',
                               title='Usage Clusters by Temperature and Rentals',
                               labels={'temp': 'Normalized Temperature', 'cnt': 'Daily Rentals'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("📊 Cluster Characteristics")
            cluster_stats = clustered_data.groupby('weather_usage_cluster').agg({
                'cnt': ['count', 'mean', 'std'],
                'temp': 'mean',
                'hum': 'mean',
                'windspeed': 'mean'
            }).round(2)
            st.dataframe(cluster_stats)
        
        with tab3:
            st.subheader("🏆 Performance Segmentation Analysis")
            
            # Create performance segments
            day_filtered['performance_segment'] = pd.cut(
                day_filtered['cnt'], 
                bins=3, 
                labels=['Low Performance', 'Medium Performance', 'High Performance']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                perf_by_season = pd.crosstab(day_filtered['season_name'], 
                                           day_filtered['performance_segment'], 
                                           normalize='index') * 100
                
                fig = px.bar(perf_by_season.T, 
                            title='Performance Distribution by Season (%)',
                            labels={'index': 'Performance Level', 'value': 'Percentage'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                perf_by_weather = pd.crosstab(day_filtered['weathersit_name'], 
                                            day_filtered['performance_segment'], 
                                            normalize='index') * 100
                
                fig = px.bar(perf_by_weather.T, 
                            title='Performance Distribution by Weather (%)',
                            labels={'index': 'Performance Level', 'value': 'Percentage'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Conclusions and Recommendations
    st.markdown("---")
    st.header("💡 Key Insights and Business Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Key Insights</h4>
        <p><strong>Seasonal Patterns:</strong></p>
        <ul>
        <li>Fall shows highest average usage, followed by Summer</li>
        <li>Winter has the lowest bike sharing demand</li>
        <li>Clear weather conditions significantly boost usage</li>
        </ul>
        <p><strong>Temporal Patterns:</strong></p>
        <ul>
        <li>Weekday usage peaks at 8 AM and 5-6 PM (commuting hours)</li>
        <li>Weekend usage is more evenly distributed throughout the day</li>
        <li>Hour 17 (5 PM) shows highest overall usage</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="recommendation-box">
        <h4>🚀 Business Recommendations</h4>
        <p><strong>Operations:</strong></p>
        <ul>
        <li>Increase bike availability during peak commuting hours</li>
        <li>Implement dynamic pricing based on weather conditions</li>
        <li>Schedule maintenance during low-demand winter periods</li>
        </ul>
        <p><strong>Marketing:</strong></p>
        <ul>
        <li>Promote weekend recreational usage with special packages</li>
        <li>Target casual users for conversion to registered users</li>
        <li>Weather-based promotional campaigns during clear days</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer


if __name__ == "__main__":
    main()