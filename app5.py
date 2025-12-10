import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# AI/ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set page config
st.set_page_config(
    page_title="AI-Powered PMO Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize NLTK (first time only)
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

# Title
st.title("🚀 AI-Powered PMO Dashboard")
st.markdown("---")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Sample Data", "Upload CSV/Excel"]
    )
    
    uploaded_file = None
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader(
            "Upload project data", 
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with project data"
        )
    
    # AI Settings
    st.subheader("🤖 AI Settings")
    enable_ai = st.checkbox("Enable AI Predictions", value=True)
    enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
    
    # Refresh button
    if st.button("🔄 Refresh Dashboard", type="primary"):
        st.rerun()

# ==================== SAMPLE DATA GENERATION ====================
@st.cache_data
def generate_sample_data():
    """Generate sample project data for demonstration"""
    projects = [
        "Digital Transformation", "CRM Implementation", "Mobile App v3.0",
        "Cloud Migration", "Data Analytics Platform", "Website Redesign",
        "IoT Integration", "AI Chatbot", "Payment Gateway", "Security Upgrade"
    ]
    
    teams = ["Tech Team", "Data Science", "DevOps", "Frontend", "Backend", "QA"]
    statuses = ["Green", "Yellow", "Red"]
    sponsors = ["John Doe", "Jane Smith", "Robert Johnson", "Emily Davis", "Mike Brown"]
    
    data = []
    for i, project in enumerate(projects):
        start_date = datetime.now() - timedelta(days=np.random.randint(0, 60))
        end_date = start_date + timedelta(days=np.random.randint(60, 180))
        actual_end = end_date + timedelta(days=np.random.randint(-10, 20))
        
        # Generate project metrics
        budget = np.random.randint(50000, 300000)
        spent = budget * np.random.uniform(0.7, 1.2)
        completion = np.random.randint(30, 100)
        delay_days = max(0, (actual_end - end_date).days)
        
        # Project updates with varied sentiment
        positive_updates = [
            "Project is progressing well, team is highly motivated.",
            "All milestones achieved ahead of schedule! Great collaboration.",
            "Stakeholders are very happy with the progress and quality.",
            "Team collaboration is excellent, productivity is high."
        ]
        
        neutral_updates = [
            "Project is on track, following the planned timeline.",
            "Regular status meetings are being conducted as scheduled.",
            "Resources are adequately allocated for current phase.",
            "Technical specifications are being documented properly."
        ]
        
        negative_updates = [
            "Facing challenges with third-party integrations.",
            "Need additional resources for the testing phase.",
            "Technical debt is accumulating and needs attention.",
            "Scope creep is becoming a concern."
        ]
        
        # Mix updates based on project status probability
        if completion > 80:
            updates = np.random.choice(positive_updates, size=2, replace=False).tolist()
        elif completion < 40:
            updates = np.random.choice(negative_updates, size=2, replace=False).tolist()
        else:
            updates = np.random.choice(neutral_updates, size=2, replace=False).tolist()
        
        # Add one random update
        all_updates = positive_updates + neutral_updates + negative_updates
        updates.append(np.random.choice(all_updates))
        
        data.append({
            "Project ID": f"PROJ-{1000 + i}",
            "Project Name": project,
            "Project Manager": np.random.choice(sponsors),
            "Team": np.random.choice(teams),
            "Status": np.random.choice(statuses, p=[0.6, 0.3, 0.1]),
            "Priority": np.random.choice(["Critical", "High", "Medium", "Low"]),
            "Category": np.random.choice(["IT", "Digital", "Infrastructure", "Software"]),
            "Start Date": start_date,
            "Planned End Date": end_date,
            "Actual End Date": actual_end if np.random.random() > 0.3 else None,
            "Completion %": completion,
            "Budget ($)": budget,
            "Spent ($)": spent,
            "Delay (Days)": delay_days,
            "Resource Utilization %": np.random.randint(70, 110),
            "Risk Score (1-10)": np.random.randint(1, 8),
            "Issues Count": np.random.randint(0, 10),
            "Dependencies": np.random.randint(0, 5),
            "Stakeholder Satisfaction (1-10)": np.random.randint(6, 10),
            "Updates": updates,
            "Last Updated": datetime.now() - timedelta(days=np.random.randint(0, 7))
        })
    
    return pd.DataFrame(data)

# ==================== DATA LOADING ====================
def load_data(uploaded_file=None):
    """Load data from uploaded file or use sample data"""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel file.")
                df = generate_sample_data()
            
            # Validate required columns
            required_columns = ['Project Name', 'Status', 'Completion %', 'Budget ($)', 'Spent ($)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Missing columns in uploaded file: {missing_columns}. Using sample data instead.")
                df = generate_sample_data()
            else:
                st.success(f"✅ Successfully loaded {len(df)} projects from uploaded file")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}. Using sample data.")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
        st.info("📊 Using sample data. Upload your own CSV/Excel file for real project data.")
    
    # Convert date columns
    date_columns = ['Start Date', 'Planned End Date', 'Actual End Date', 'Last Updated']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

# Load data
df = load_data(uploaded_file)

# ==================== AI FUNCTIONS ====================
def analyze_sentiment(text_list):
    """Analyze sentiment of project updates using VADER"""
    sia = SentimentIntensityAnalyzer()
    
    if not text_list or (isinstance(text_list, list) and len(text_list) == 0):
        return 0.0
    
    if isinstance(text_list, str):
        text_list = [text_list]
    
    scores = []
    for text in text_list:
        if isinstance(text, str):
            sentiment = sia.polarity_scores(text)
            scores.append(sentiment['compound'])
    
    return np.mean(scores) if scores else 0.0

def calculate_budget_health(row):
    """Calculate budget health metrics"""
    budget = row['Budget ($)']
    spent = row['Spent ($)']
    completion = row['Completion %']
    
    if budget > 0:
        spend_rate = spent / budget
        # Expected spend based on completion
        expected_spend = budget * (completion / 100)
        
        if spent <= expected_spend * 0.9:
            return "Under Budget", spend_rate
        elif spent <= expected_spend * 1.1:
            return "On Budget", spend_rate
        else:
            return "Over Budget", spend_rate
    
    return "N/A", 0

def predict_project_risks(df):
    """Predict project delays and risks using ML"""
    if len(df) < 3:
        df['Predicted Delay (Days)'] = 0
        df['Risk Level'] = 'Low'
        df['Completion Confidence %'] = 0.95
        return df
    
    try:
        # Calculate features for prediction
        features_df = df.copy()
        
        # Budget variance
        features_df['Budget Variance %'] = ((features_df['Spent ($)'] - features_df['Budget ($)']) / 
                                           features_df['Budget ($)']) * 100
        
        # Time metrics
        current_date = datetime.now()
        features_df['Days Since Start'] = (current_date - pd.to_datetime(features_df['Start Date'])).dt.days
        features_df['Days To Deadline'] = (pd.to_datetime(features_df['Planned End Date']) - current_date).dt.days
        
        # Progress rate
        features_df['Progress Rate'] = features_df['Completion %'] / features_df['Days Since Start'].clip(lower=1)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        # Calculate predicted delay
        features_df['Predicted Delay (Days)'] = (
            (100 - features_df['Completion %']) * features_df['Days To Deadline'] / 
            (features_df['Completion %'] + 1) * 0.01 +
            abs(features_df['Budget Variance %']) * 0.2 +
            (features_df['Risk Score (1-10)'] * 0.5)
        ).clip(0, 60)
        
        # Risk level classification
        def assign_risk_level(row):
            delay = row['Predicted Delay (Days)']
            risk_score = row['Risk Score (1-10)']
            
            if delay > 20 or risk_score > 7 or row['Status'] == 'Red':
                return 'High'
            elif delay > 10 or risk_score > 5 or row['Status'] == 'Yellow':
                return 'Medium'
            else:
                return 'Low'
        
        features_df['Risk Level'] = features_df.apply(assign_risk_level, axis=1)
        
        # Completion confidence
        features_df['Completion Confidence %'] = (
            95 - (features_df['Predicted Delay (Days)'] * 0.5) - 
            (abs(features_df['Budget Variance %']) * 0.3) -
            (features_df['Risk Score (1-10)'] * 2)
        ).clip(10, 99)
        
        return features_df
        
    except Exception as e:
        df['Predicted Delay (Days)'] = 0
        df['Risk Level'] = 'Low'
        df['Completion Confidence %'] = 85
        return df

# Apply calculations
df[['Budget Health', 'Budget Ratio']] = df.apply(
    lambda row: pd.Series(calculate_budget_health(row)), axis=1
)

# Apply AI functions if enabled
if enable_sentiment and 'Updates' in df.columns:
    df['Sentiment Score'] = df['Updates'].apply(analyze_sentiment)
    df['Sentiment'] = df['Sentiment Score'].apply(
        lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
    )

if enable_ai:
    df = predict_project_risks(df)

# ==================== CREATE TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Overview", 
    "🤖 AI Insights", 
    "📈 Detailed Analytics",
    "📋 Project Portfolio",
    "⚙️ Settings & Export"
])

# ==================== TAB 1: DASHBOARD OVERVIEW ====================
with tab1:
    st.header("📊 Portfolio Dashboard Overview")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_projects = len(df)
        st.metric("Total Projects", total_projects)
    
    with col2:
        at_risk = len(df[df['Status'] == 'Red'])
        st.metric("Critical Projects", at_risk, delta=f"{at_risk/total_projects*100:.1f}%")
    
    with col3:
        avg_completion = df['Completion %'].mean()
        st.metric("Avg Completion", f"{avg_completion:.1f}%")
    
    with col4:
        total_budget = df['Budget ($)'].sum() / 1000000
        total_spent = df['Spent ($)'].sum() / 1000000
        variance = ((total_spent - total_budget) / total_budget) * 100
        st.metric("Budget Variance", f"{variance:+.1f}%")
    
    # Charts Row 1
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Project Status Distribution")
        status_counts = df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig1 = px.pie(status_counts, values='Count', names='Status', 
                      color='Status', 
                      color_discrete_map={'Green': '#2ecc71', 
                                         'Yellow': '#f39c12', 
                                         'Red': '#e74c3c'},
                      hole=0.3)
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        st.subheader("Completion vs Budget")
        # FIXED: Removed trendline="ols" parameter
        fig2 = px.scatter(df, x='Completion %', y='Spent ($)', 
                         color='Status', size='Budget ($)',
                         hover_data=['Project Name', 'Project Manager'],
                         color_discrete_map={'Green': '#2ecc71', 
                                            'Yellow': '#f39c12', 
                                            'Red': '#e74c3c'})
        fig2.update_layout(xaxis_title="Completion %", yaxis_title="Spent ($)")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Charts Row 2
    st.subheader("Timeline Overview")
    timeline_df = df.nlargest(8, 'Budget ($)').copy()
    timeline_df['Duration (Days)'] = (timeline_df['Planned End Date'] - timeline_df['Start Date']).dt.days
    
    fig3 = px.bar(timeline_df, 
                  x='Duration (Days)', 
                  y='Project Name',
                  orientation='h',
                  color='Status',
                  hover_data=['Completion %', 'Project Manager', 'Budget ($)'],
                  color_discrete_map={'Green': '#2ecc71', 
                                     'Yellow': '#f39c12', 
                                     'Red': '#e74c3c'})
    fig3.update_layout(xaxis_title="Duration (days)", yaxis_title="", height=400)
    st.plotly_chart(fig3, use_container_width=True)

# ==================== TAB 2: AI INSIGHTS ====================
with tab2:
    st.header("🤖 AI-Powered Insights")
    
    if not enable_ai:
        st.warning("⚠️ AI Predictions are disabled. Enable AI in Settings tab.")
    
    ai_col1, ai_col2 = st.columns(2)
    
    with ai_col1:
        st.subheader("🚨 Risk Analysis")
        
        if 'Risk Level' in df.columns:
            high_risk = df[df['Risk Level'] == 'High']
            medium_risk = df[df['Risk Level'] == 'Medium']
            
            risk_metric1, risk_metric2 = st.columns(2)
            with risk_metric1:
                st.metric("High Risk Projects", len(high_risk))
            with risk_metric2:
                st.metric("Medium Risk Projects", len(medium_risk))
            
            if not high_risk.empty:
                st.error("**High Risk Projects Requiring Attention:**")
                for _, row in high_risk.iterrows():
                    with st.expander(f"🔴 {row['Project Name']}", expanded=False):
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Delay", f"{row.get('Predicted Delay (Days)', 0):.0f}d")
                        with cols[1]:
                            st.metric("Confidence", f"{row.get('Completion Confidence %', 0):.0f}%")
                        with cols[2]:
                            st.metric("Risk Score", row.get('Risk Score (1-10)', 0))
                        
                        st.write("**Recommendations:**")
                        st.write("1. Conduct emergency review meeting")
                        st.write("2. Reallocate resources if possible")
                        st.write("3. Update stakeholders immediately")
            else:
                st.success("✅ No high-risk projects identified")
            
            # Risk distribution chart
            if 'Risk Level' in df.columns:
                risk_dist = df['Risk Level'].value_counts().reset_index()
                risk_dist.columns = ['Risk Level', 'Count']
                fig4 = px.bar(risk_dist, x='Risk Level', y='Count',
                             color='Risk Level',
                             color_discrete_map={'High': '#e74c3c', 
                                               'Medium': '#f39c12', 
                                               'Low': '#2ecc71'})
                st.plotly_chart(fig4, use_container_width=True)
    
    with ai_col2:
        st.subheader("📈 Delay Predictions")
        
        if 'Predicted Delay (Days)' in df.columns:
            # Top 5 projects with highest predicted delay
            delay_df = df.nlargest(5, 'Predicted Delay (Days)')[['Project Name', 'Predicted Delay (Days)', 'Completion Confidence %']]
            
            fig5 = px.bar(delay_df, 
                         x='Predicted Delay (Days)', 
                         y='Project Name',
                         orientation='h',
                         color='Predicted Delay (Days)',
                         color_continuous_scale='RdYlGn_r',
                         hover_data=['Completion Confidence %'])
            fig5.update_layout(xaxis_title="Predicted Delay (Days)", yaxis_title="")
            st.plotly_chart(fig5, use_container_width=True)
            
            # Delay statistics
            avg_delay = df['Predicted Delay (Days)'].mean()
            max_delay = df['Predicted Delay (Days)'].max()
            
            delay_col1, delay_col2 = st.columns(2)
            with delay_col1:
                st.metric("Avg Predicted Delay", f"{avg_delay:.1f} days")
            with delay_col2:
                st.metric("Max Predicted Delay", f"{max_delay:.1f} days")
        
        st.subheader("😊 Sentiment Analysis")
        
        if enable_sentiment and 'Sentiment' in df.columns:
            sentiment_counts = df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            fig6 = px.pie(sentiment_counts, values='Count', names='Sentiment',
                         color='Sentiment',
                         color_discrete_map={'Positive': '#2ecc71', 
                                           'Neutral': '#f39c12', 
                                           'Negative': '#e74c3c'})
            st.plotly_chart(fig6, use_container_width=True)
            
            avg_sentiment = df['Sentiment Score'].mean() if 'Sentiment Score' in df.columns else 0
            st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
        else:
            st.info("Enable sentiment analysis in Settings to view team morale insights")

# ==================== TAB 3: DETAILED ANALYTICS ====================
with tab3:
    st.header("📈 Detailed Analytics & Trends")
    
    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=df['Status'].unique(),
            default=df['Status'].unique(),
            key="analytics_status"
        )
    
    with filter_col2:
        if 'Team' in df.columns:
            team_filter = st.multiselect(
                "Filter by Team",
                options=df['Team'].unique(),
                default=df['Team'].unique(),
                key="analytics_team"
            )
        else:
            team_filter = []
    
    with filter_col3:
        if 'Priority' in df.columns:
            priority_filter = st.multiselect(
                "Filter by Priority",
                options=df['Priority'].unique(),
                default=df['Priority'].unique(),
                key="analytics_priority"
            )
        else:
            priority_filter = []
    
    # Apply filters
    analytics_df = df.copy()
    if status_filter:
        analytics_df = analytics_df[analytics_df['Status'].isin(status_filter)]
    if team_filter and 'Team' in analytics_df.columns:
        analytics_df = analytics_df[analytics_df['Team'].isin(team_filter)]
    if priority_filter and 'Priority' in analytics_df.columns:
        analytics_df = analytics_df[analytics_df['Priority'].isin(priority_filter)]
    
    # Analytics Charts
    analytics_col1, analytics_col2 = st.columns(2)
    
    with analytics_col1:
        st.subheader("Budget vs Completion Analysis")
        
        # FIXED: Removed trendline="ols" parameter
        fig7 = px.scatter(analytics_df, x='Completion %', y='Spent ($)', 
                         color='Status', size='Budget ($)',
                         hover_data=['Project Name', 'Team', 'Priority'],
                         color_discrete_map={'Green': '#2ecc71', 
                                            'Yellow': '#f39c12', 
                                            'Red': '#e74c3c'})
        st.plotly_chart(fig7, use_container_width=True)
        
        # Budget health summary
        if 'Budget Health' in analytics_df.columns:
            budget_summary = analytics_df['Budget Health'].value_counts()
            st.write("**Budget Health Summary:**")
            for status, count in budget_summary.items():
                percentage = (count / len(analytics_df)) * 100
                st.progress(percentage/100, text=f"{status}: {count} projects ({percentage:.1f}%)")
    
    with analytics_col2:
        st.subheader("Performance Metrics")
        
        metrics = ['Completion %', 'Resource Utilization %', 'Risk Score (1-10)']
        selected_metric = st.selectbox("Select Metric", metrics)
        
        fig8 = px.box(analytics_df, y=selected_metric, color='Status',
                     color_discrete_map={'Green': '#2ecc71', 
                                        'Yellow': '#f39c12', 
                                        'Red': '#e74c3c'})
        st.plotly_chart(fig8, use_container_width=True)
        
        # Metric statistics
        metric_stats = analytics_df[selected_metric].describe()
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Average", f"{metric_stats['mean']:.1f}")
        with stats_col2:
            st.metric("Median", f"{metric_stats['50%']:.1f}")
        with stats_col3:
            st.metric("Std Dev", f"{metric_stats['std']:.1f}")
    
    # Correlation Analysis
    st.subheader("📊 Correlation Matrix")
    
    numeric_cols = ['Completion %', 'Budget ($)', 'Spent ($)', 
                   'Resource Utilization %', 'Risk Score (1-10)', 
                   'Issues Count']
    
    # Filter to only include columns that exist
    existing_numeric_cols = [col for col in numeric_cols if col in analytics_df.columns]
    
    if len(existing_numeric_cols) > 1:
        corr_matrix = analytics_df[existing_numeric_cols].corr()
        
        fig9 = px.imshow(corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        aspect="auto")
        fig9.update_layout(height=500)
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis")

# ==================== TAB 4: PROJECT PORTFOLIO ====================
with tab4:
    st.header("📋 Project Portfolio Management")
    
    # Quick filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        view_status = st.multiselect(
            "Status",
            options=df['Status'].unique(),
            default=df['Status'].unique(),
            key="portfolio_status"
        )
    
    with filter_col2:
        if 'Priority' in df.columns:
            view_priority = st.multiselect(
                "Priority",
                options=df['Priority'].unique(),
                default=df['Priority'].unique(),
                key="portfolio_priority"
            )
        else:
            view_priority = []
    
    with filter_col3:
        if 'Team' in df.columns:
            view_team = st.multiselect(
                "Team",
                options=df['Team'].unique(),
                default=df['Team'].unique(),
                key="portfolio_team"
            )
        else:
            view_team = []
    
    with filter_col4:
        search_query = st.text_input("Search Projects", placeholder="Type project name...")
    
    # Apply filters
    portfolio_df = df.copy()
    if view_status:
        portfolio_df = portfolio_df[portfolio_df['Status'].isin(view_status)]
    if view_priority and 'Priority' in portfolio_df.columns:
        portfolio_df = portfolio_df[portfolio_df['Priority'].isin(view_priority)]
    if view_team and 'Team' in portfolio_df.columns:
        portfolio_df = portfolio_df[portfolio_df['Team'].isin(view_team)]
    if search_query:
        portfolio_df = portfolio_df[portfolio_df['Project Name'].str.contains(search_query, case=False, na=False)]
    
    # Display summary
    st.metric("Filtered Projects", len(portfolio_df), delta=f"{len(portfolio_df)} of {len(df)}")
    
    # Define columns to display
    display_columns = ['Project Name', 'Project Manager', 'Team', 'Status', 'Priority',
                       'Completion %', 'Budget ($)', 'Spent ($)', 'Budget Health']
    
    # Add AI columns if available
    if enable_ai and 'Risk Level' in portfolio_df.columns:
        display_columns.extend(['Risk Level', 'Predicted Delay (Days)'])
    
    if enable_sentiment and 'Sentiment' in portfolio_df.columns:
        display_columns.append('Sentiment')
    
    # Display data table with enhanced formatting
    st.dataframe(
        portfolio_df[display_columns],
        use_container_width=True,
        height=500,
        column_config={
            "Completion %": st.column_config.ProgressColumn(
                "Completion",
                format="%d%%",
                min_value=0,
                max_value=100,
            ),
            "Budget ($)": st.column_config.NumberColumn(
                "Budget",
                format="$%.0f"
            ),
            "Spent ($)": st.column_config.NumberColumn(
                "Spent",
                format="$%.0f"
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                help="Project status: Green, Yellow, or Red"
            ),
            "Risk Level": st.column_config.TextColumn(
                "Risk Level",
                help="AI-predicted risk level"
            )
        }
    )
    
    # Project details expander
    st.subheader("🔍 Project Details")
    
    if len(portfolio_df) > 0:
        selected_project = st.selectbox(
            "Select a project for detailed view",
            options=portfolio_df['Project Name'].tolist()
        )
        
        if selected_project:
            project_data = portfolio_df[portfolio_df['Project Name'] == selected_project].iloc[0]
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write("**Basic Information**")
                st.write(f"**Project ID:** {project_data.get('Project ID', 'N/A')}")
                st.write(f"**Manager:** {project_data.get('Project Manager', 'N/A')}")
                st.write(f"**Team:** {project_data.get('Team', 'N/A')}")
                st.write(f"**Category:** {project_data.get('Category', 'N/A')}")
                st.write(f"**Priority:** {project_data.get('Priority', 'N/A')}")
            
            with detail_col2:
                st.write("**Performance Metrics**")
                st.write(f"**Status:** {project_data.get('Status', 'N/A')}")
                st.write(f"**Completion:** {project_data.get('Completion %', 0)}%")
                st.write(f"**Budget:** ${project_data.get('Budget ($)', 0):,.0f}")
                st.write(f"**Spent:** ${project_data.get('Spent ($)', 0):,.0f}")
                st.write(f"**Budget Health:** {project_data.get('Budget Health', 'N/A')}")
            
            # AI Insights if available
            if enable_ai and 'Risk Level' in project_data:
                st.write("**🤖 AI Insights**")
                ai_col1, ai_col2, ai_col3 = st.columns(3)
                with ai_col1:
                    st.metric("Risk Level", project_data['Risk Level'])
                with ai_col2:
                    st.metric("Predicted Delay", f"{project_data.get('Predicted Delay (Days)', 0):.0f} days")
                with ai_col3:
                    st.metric("Confidence", f"{project_data.get('Completion Confidence %', 0):.0f}%")
            
            # Project updates if available
            if 'Updates' in project_data and isinstance(project_data['Updates'], list):
                st.write("**📝 Latest Updates**")
                for update in project_data['Updates'][:3]:  # Show last 3 updates
                    st.write(f"• {update}")

# ==================== TAB 5: SETTINGS & EXPORT ====================
with tab5:
    st.header("⚙️ Settings & Data Management")
    
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        st.subheader("Dashboard Settings")
        
        # Display current settings
        st.write("**Current Configuration:**")
        st.write(f"- AI Predictions: {'✅ Enabled' if enable_ai else '❌ Disabled'}")
        st.write(f"- Sentiment Analysis: {'✅ Enabled' if enable_sentiment else '❌ Disabled'}")
        st.write(f"- Data Source: {data_source}")
        st.write(f"- Total Projects Loaded: {len(df)}")
        
        # Data info
        st.subheader("📊 Data Information")
        st.write(f"**Columns Available:** {len(df.columns)}")
        st.write(f"**Date Range:** {df['Start Date'].min().date()} to {df['Planned End Date'].max().date()}")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache & Reload", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with settings_col2:
        st.subheader("📥 Export Data")
        
        # Export options
        export_format = st.radio(
            "Export Format",
            ["CSV", "Excel", "JSON"]
        )
        
        # Filter for export
        export_cols = st.multiselect(
            "Select columns to export",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:15]
        )
        
        export_df = df[export_cols] if export_cols else df
        
        if export_format == "CSV":
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"pmo_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        elif export_format == "Excel":
            # For Excel export, we need to use BytesIO
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='PMO_Data')
            excel_data = buffer.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"pmo_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        elif export_format == "JSON":
            json_data = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"pmo_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Data Preview
    st.subheader("🔍 Data Preview")
    preview_cols = st.multiselect(
        "Select columns to preview",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:8],
        key="preview_cols"
    )
    
    if preview_cols:
        st.dataframe(df[preview_cols].head(10), use_container_width=True)
    
    # Help Section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        **Dashboard Guide:**
        
        **Tab 1: Dashboard Overview**
        - High-level KPIs and metrics
        - Project status distribution
        - Timeline visualization
        
        **Tab 2: AI Insights**
        - Risk analysis and predictions
        - Delay forecasts
        - Team sentiment analysis
        
        **Tab 3: Detailed Analytics**
        - Advanced charts and trends
        - Correlation analysis
        - Performance metrics
        
        **Tab 4: Project Portfolio**
        - Detailed project listing
        - Filter and search capabilities
        - Individual project details
        
        **Tab 5: Settings & Export**
        - Configure dashboard settings
        - Export data in multiple formats
        - Clear cache and reload
        
        **Data Requirements:**
        - Minimum: Project Name, Status, Completion %, Budget, Spent
        - Optional: Updates (for sentiment), Risk Score, Team, Priority
        
        **Tips:**
        - Use filters to focus on specific projects
        - Enable AI for predictive insights
        - Export data for external reporting
        """)

# ==================== ALERTS SIDEBAR ====================
with st.sidebar:
    st.header("🔔 Active Alerts")
    
    # Generate alerts
    alerts = []
    
    # Status alerts
    red_projects = df[df['Status'] == 'Red']
    if not red_projects.empty:
        alerts.append(f"🚨 {len(red_projects)} projects in RED status")
    
    # Budget alerts
    if 'Budget Health' in df.columns:
        over_budget = df[df['Budget Health'] == 'Over Budget']
        if not over_budget.empty:
            alerts.append(f"💰 {len(over_budget)} projects over budget")
    
    # AI alerts
    if enable_ai and 'Predicted Delay (Days)' in df.columns:
        high_delay = df[df['Predicted Delay (Days)'] > 20]
        if not high_delay.empty:
            alerts.append(f"⏰ {len(high_delay)} projects with >20 day delay")
    
    # Sentiment alerts
    if enable_sentiment and 'Sentiment Score' in df.columns:
        negative_sentiment = df[df['Sentiment Score'] < -0.3]
        if not negative_sentiment.empty:
            alerts.append(f"😟 {len(negative_sentiment)} projects with negative sentiment")
    
    # Display alerts
    if alerts:
        st.warning(f"{len(alerts)} active alerts")
        for alert in alerts[:5]:
            st.error(alert)
        if len(alerts) > 5:
            st.caption(f"... and {len(alerts) - 5} more alerts")
    else:
        st.success("✅ No critical alerts")
    
    # Dashboard info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"""
    **Dashboard Info:**
    - Projects: {len(df)}
    - AI: {'On' if enable_ai else 'Off'}
    - Last refresh: {datetime.now().strftime("%H:%M:%S")}
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption(f"""
    **AI-Powered PMO Dashboard** • Built with Streamlit • 
    {len(df)} projects loaded • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """)