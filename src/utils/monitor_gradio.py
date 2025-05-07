import pandas as pd
import yaml
import platform
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text
from sqlalchemy import create_engine
import sqlalchemy
import gradio as gr
import numpy as np
import sys
import os

# Configuration setup
def get_from_cnfg(key_path, file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        keys = key_path.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

# Environment setup
os_name = platform.system()
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    from google.colab import userdata
    engine = create_engine(userdata.get('DB_URL'))
    source_folder='/content/drive/MyDrive/Health_Data/MIMIC_JPG_AVL/mimic-cxr-jpg/2.1.0/files/'
elif os_name == "Darwin":
    cnfig_file="/Users/bineshkumar/Documents/config.yaml"
    DB_URL = get_from_cnfg("cd_url", cnfig_file)
    source_folder='/Users/bineshkumar/Documents/mimic-cxr-jpg/2.1.0/files/'
elif os_name == "Linux":
    cnfig_file="/Users/bineshkumar/Documents/config.yaml"
    DB_URL = get_from_cnfg("cd_url", cnfig_file)
    source_folder=""

# Disable SQLAlchemy version detection warning
import warnings
from sqlalchemy import exc as sa_exc
warnings.filterwarnings('ignore', category=sa_exc.SAWarning)

# Make a simple connection function that doesn't rely on SQLAlchemy
def get_db_connection():
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Initialize database connection
conn = get_db_connection()
if conn:
    print("Database connection established")
    engine = conn
else:
    print("Failed to connect to database")
    engine = None

# Data retrieval functions
def get_hourly_model_stats():
    if engine is None:
        # Return empty dataframe if no DB connection
        return pd.DataFrame(columns=['model_name', 'hour', 'unique_count'])
    
    # Use psycopg2 directly instead of SQLAlchemy
    query = """
    SELECT
        model_name,
        date_trunc('hour', created_at) as hour,
        COUNT(*) as unique_count
    FROM
        public.model_responses_r2
    GROUP BY
        model_name,
        date_trunc('hour', created_at)
    ORDER BY
        hour ASC,
        model_name
    """

    try:
        # Using psycopg2 connection directly
        cursor = engine.cursor()
        cursor.execute(query)
        
        # Convert results to DataFrame
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        
        # Convert hour to datetime
        if 'hour' in df.columns and not df.empty:
            df['hour'] = pd.to_datetime(df['hour'])
            
        cursor.close()
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(columns=['model_name', 'hour', 'unique_count'])

def get_model_cumulative_trend(model_names=None, days=30):
    if engine is None:
        # Return empty dataframe if no DB connection
        return pd.DataFrame(columns=['model_name', 'hour', 'unique_count'])
        
    try:
        # Construct WHERE clause for model names
        if model_names and len(model_names) > 0:
            placeholders = ','.join([f"'{model}'" for model in model_names])
            where_clause = f"WHERE model_name IN ({placeholders})"
        else:
            where_clause = "WHERE 1=1"  # Always true condition to simplify query structure

        # Build the query string with days parameter
        query = f"""
        SELECT
            model_name,
            date_trunc('hour', created_at) as hour,
            COUNT(*) as unique_count
        FROM
            public.model_responses_r2
        {where_clause}
        AND created_at >= NOW() - INTERVAL '{days} days'
        GROUP BY
            model_name,
            date_trunc('hour', created_at)
        ORDER BY
            hour
        """

        # Using psycopg2 connection directly
        cursor = engine.cursor()
        cursor.execute(query)
        
        # Convert results to DataFrame
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        
        # Convert hour to datetime
        if 'hour' in df.columns and not df.empty:
            df['hour'] = pd.to_datetime(df['hour'])
            if df['hour'].dt.tz is None:
                df['hour'] = df['hour'].dt.tz_localize('UTC')
                
        cursor.close()
        return df
            
    except Exception as e:
        print(f"Error in cumulative trend: {e}")
        return pd.DataFrame(columns=['model_name', 'hour', 'unique_count'])

# Visualization functions
def create_visualizations(stats_df):
    if stats_df.empty:
        # Return empty figures if no data
        fig1, fig2, fig3 = plt.figure(), plt.figure(), plt.figure()
        for fig in [fig1, fig2, fig3]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", 
                   horizontalalignment='center', verticalalignment='center')
        return fig1, fig2, fig3
    
    # Convert timestamps to UTC to ensure consistency
    if 'hour' in stats_df.columns:
        if not pd.api.types.is_datetime64_dtype(stats_df['hour']):
            stats_df['hour'] = pd.to_datetime(stats_df['hour'])

        # If timestamps don't have timezone info, localize them to UTC
        if stats_df['hour'].dt.tz is None:
            stats_df['hour'] = stats_df['hour'].dt.tz_localize('UTC')

        # Create pivot table
        pivot_df = stats_df.pivot_table(
            index='hour',
            columns='model_name',
            values='unique_count',
            aggfunc='sum'
        ).fillna(0)

        # Figure 1: Hourly metrics
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        if not pivot_df.empty:
            # Get the last 24 hours of data
            current_time_utc = pd.Timestamp.now(tz='UTC')
            day_ago_utc = current_time_utc - pd.Timedelta(days=1)
            recent_pivot = pivot_df[pivot_df.index >= day_ago_utc]

            if not recent_pivot.empty:
                recent_pivot.plot(kind='line', marker='o', ax=ax1)
                ax1.set_title('Hourly UIDs by Model - Last 24 Hours')
                ax1.set_xlabel('Hour')
                ax1.set_ylabel('Unique Combinations (uid-model-question)')
                ax1.grid(True)
            else:
                ax1.text(0.5, 0.5, "No data available for last 24 hours",
                       horizontalalignment='center', verticalalignment='center')
        else:
            ax1.text(0.5, 0.5, "No data available",
                   horizontalalignment='center', verticalalignment='center')
                   
        # Figure 2: Cumulative metrics
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        if not pivot_df.empty:
            cumulative_df = pivot_df.cumsum()
            cumulative_df.plot(kind='line', ax=ax2)
            ax2.set_title('Cumulative UIDs by Model')
            ax2.set_xlabel('Hour')
            ax2.set_ylabel('Cumulative Unique Combinations')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "No data available",
                   horizontalalignment='center', verticalalignment='center')
                   
        # Figure 3: Heatmap
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        
        if not pivot_df.empty:
            current_time_utc = pd.Timestamp.now(tz='UTC')
            day_ago_utc = current_time_utc - pd.Timedelta(days=1)
            recent_pivot = pivot_df[pivot_df.index >= day_ago_utc]
            
            if not recent_pivot.empty:
                sns.heatmap(recent_pivot.T, cmap="YlGnBu", annot=True, fmt="g", ax=ax3)
                ax3.set_title('Model Activity Heatmap - Last 24 Hours')
            else:
                ax3.text(0.5, 0.5, "No data available for heatmap",
                       horizontalalignment='center', verticalalignment='center')
        else:
            ax3.text(0.5, 0.5, "No data available",
                   horizontalalignment='center', verticalalignment='center')
        
        return fig1, fig2, fig3

# Gradio interface functions
def update_dashboard():
    stats_df = get_hourly_model_stats()
    
    # Calculate stats
    stats_html = ""
    if not stats_df.empty:
        # Most recent hour stats
        most_recent_hour = stats_df['hour'].max()
        recent_stats = stats_df[stats_df['hour'] == most_recent_hour]
        
        # Calculate cumulative totals
        cumulative_stats = stats_df.pivot_table(
            index='model_name',
            values='unique_count',
            aggfunc='sum'
        ).reset_index().sort_values('unique_count', ascending=False)
        
        # Format stats as HTML
        stats_html += f"<h3>Model Stats for {most_recent_hour}</h3>"
        stats_html += recent_stats[['model_name', 'unique_count']].sort_values('unique_count', ascending=False).to_html(index=False)
        
        stats_html += "<h3>Cumulative Stats (All Time)</h3>"
        stats_html += cumulative_stats.to_html(index=False)
        
        # Calculate total UIDs
        total_recent = recent_stats['unique_count'].sum()
        total_cumulative = cumulative_stats['unique_count'].sum()
        stats_html += f"<p>Total unique combinations in the last hour: {total_recent}</p>"
        stats_html += f"<p>Total unique combinations all time: {total_cumulative}</p>"
    else:
        stats_html = "<p>No data available</p>"
    
    # Create visualizations
    hourly_fig, cumulative_fig, heatmap_fig = create_visualizations(stats_df)
    
    # Save data to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"monitor_runs_stats/model_stats_{timestamp}.csv"
    if not stats_df.empty:
        os.makedirs("monitor_runs_stats", exist_ok=True)
        stats_df.to_csv(csv_path, index=False)
        csv_message = f"Data saved to {csv_path}"
    else:
        csv_message = "No data to save"
    
    return stats_html, hourly_fig, cumulative_fig, heatmap_fig, csv_message

def refresh_data():
    return update_dashboard()

def get_model_trend(model_list, days):
    # Convert comma-separated string to list if needed
    if isinstance(model_list, str):
        models = [m.strip() for m in model_list.split(',') if m.strip()]
    else:
        models = model_list
        
    days = int(days)
    
    df = get_model_cumulative_trend(models, days)
    
    if df.empty:
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        ax1.text(0.5, 0.5, "No data available", 
                horizontalalignment='center', verticalalignment='center')
        
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        ax2.text(0.5, 0.5, "No data available", 
                horizontalalignment='center', verticalalignment='center')
        
        return fig1, fig2
    
    # Create both hourly and cumulative plots
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)

    # Pivot and plot hourly data
    pivot_df = df.pivot_table(
        index='hour',
        columns='model_name',
        values='unique_count',
        aggfunc='sum'
    ).fillna(0)

    pivot_df.plot(kind='line', marker='o', ax=ax1)
    ax1.set_title(f'Hourly Metrics - Last {days} Days')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Unique Combinations')
    ax1.grid(True)

    # Plot cumulative data
    cumulative_df = pivot_df.cumsum()
    cumulative_df.plot(kind='line', ax=ax2)
    ax2.set_title(f'Cumulative Metrics - Last {days} Days')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Cumulative Unique Combinations')
    ax2.grid(True)
    
    return fig1, fig2

# Create Gradio interface
with gr.Blocks(title="LLM Model Monitoring Dashboard") as app:
    gr.Markdown("# Real-time CockroachDB Model Monitoring")
    
    with gr.Tab("Real-time Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh Data")
                stats_output = gr.HTML()
                csv_output = gr.Textbox(label="CSV Status")
                
            with gr.Column(scale=2):
                with gr.Tab("Hourly"):
                    hourly_plot = gr.Plot()
                with gr.Tab("Cumulative"):
                    cumulative_plot = gr.Plot()
                with gr.Tab("Heatmap"):
                    heatmap_plot = gr.Plot()
    
    with gr.Tab("Specific Model Analysis"):
        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(
                    label="Model Names (comma-separated)", 
                    placeholder="e.g. llama-3-70b-instruct, gpt-4o"
                )
                days_input = gr.Slider(
                    minimum=1, 
                    maximum=90, 
                    value=30, 
                    step=1, 
                    label="Days to analyze"
                )
                analyze_btn = gr.Button("Analyze Models")
            
        with gr.Row():
            with gr.Column():
                with gr.Tab("Hourly Trend"):
                    model_hourly_plot = gr.Plot()
                with gr.Tab("Cumulative Trend"):
                    model_cumulative_plot = gr.Plot()

    # Set up event handlers
    refresh_btn.click(
        fn=refresh_data,
        outputs=[stats_output, hourly_plot, cumulative_plot, heatmap_plot, csv_output]
    )
    
    analyze_btn.click(
        fn=get_model_trend,
        inputs=[model_input, days_input],
        outputs=[model_hourly_plot, model_cumulative_plot]
    )
    
    # Initial data will be loaded when the refresh button is clicked

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)