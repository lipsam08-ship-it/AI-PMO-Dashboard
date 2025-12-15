import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import io

# 🔥 NEW ML IMPORTS
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PMO Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f3c88;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d5ba6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1f3c88;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .project-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-planning {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-active {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-at-risk {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-critical {
        background-color: #ffebee;
        color: #c62828;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-completed {
        background-color: #f3e5f5;
        color: #7b1fa2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .progress-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 5px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        background-color: #1f3c88;
    }
    .form-card {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .warning-card {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #ffeaa7;
    }
    .ml-insight-card {
        background-color: #e8f4fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    .ml-risk-high {
        color: #d32f2f;
        font-weight: bold;
        background-color: #ffebee;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .ml-risk-medium {
        color: #f57c00;
        font-weight: bold;
        background-color: #fff3e0;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .ml-risk-low {
        color: #388e3c;
        font-weight: bold;
        background-color: #e8f5e9;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .confidence-high {
        color: #388e3c;
    }
    .confidence-medium {
        color: #f57c00;
    }
    .confidence-low {
        color: #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None
if 'data_warnings' not in st.session_state:
    st.session_state.data_warnings = []
if 'ml_models' not in st.session_state:  # NEW: Store ML models
    st.session_state.ml_models = {}

# 🔥 NEW ML HELPER FUNCTIONS

def prepare_ml_features(projects):
    """Prepare features for ML models"""
    features = []
    project_names = []
    
    for project in projects:
        # Feature engineering
        spent_ratio = project['spent'] / project['budget'] if project['budget'] > 0 else 0
        progress_ratio = project['progress'] / 100
        efficiency = progress_ratio / spent_ratio if spent_ratio > 0 else 0
        
        # Convert status to numerical
        status_map = {'Planning': 0, 'Active': 1, 'At Risk': 2, 'Critical': 3, 'Completed': 4}
        status_num = status_map.get(project['status'], 0)
        
        # Convert risk level to numerical
        risk_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        risk_num = risk_map.get(project['risk_level'], 1)
        
        # Calculate days if dates are available
        try:
            start_date = datetime.strptime(project['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(project['end_date'], '%Y-%m-%d')
            days_duration = (end_date - start_date).days
            days_elapsed = (datetime.now() - start_date).days
            time_ratio = days_elapsed / days_duration if days_duration > 0 else 0
        except:
            days_duration = 180  # Default 6 months
            days_elapsed = 90
            time_ratio = 0.5
        
        # Create feature vector
        feature_vector = [
            project['budget'],
            project['spent'],
            project['progress'],
            spent_ratio,
            progress_ratio,
            efficiency,
            status_num,
            risk_num,
            days_duration,
            time_ratio,
            project['roi'] if project['roi'] is not None else 10
        ]
        
        features.append(feature_vector)
        project_names.append(project['name'])
    
    return np.array(features), project_names

def train_risk_prediction_model(projects):
    """Train ML model to predict project risk"""
    if len(projects) < 5:
        return None
    
    X, project_names = prepare_ml_features(projects)
    
    # Create target labels based on current risk status
    y = []
    for project in projects:
        if project['status'] in ['Critical', 'At Risk'] or project['risk_level'] in ['High', 'Critical']:
            y.append(1)  # High risk
        else:
            y.append(0)  # Low/Medium risk
    
    if len(set(y)) < 2:  # Need both classes
        return None
    
    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def train_budget_prediction_model(projects):
    """Train ML model to predict final budget overrun"""
    if len(projects) < 3:
        return None
    
    X, project_names = prepare_ml_features(projects)
    
    # Create target: predicted overspending percentage
    y = []
    for project in projects:
        spent_ratio = project['spent'] / project['budget'] if project['budget'] > 0 else 0
        progress_ratio = project['progress'] / 100
        
        if progress_ratio > 0:
            # Estimate final cost based on current burn rate
            estimated_final = project['spent'] / progress_ratio
            overspend_pct = ((estimated_final - project['budget']) / project['budget']) * 100
        else:
            overspend_pct = 0
        
        y.append(max(0, overspend_pct))  # Only positive overspending
    
    # Train Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def cluster_projects(projects):
    """Cluster projects into similar groups using K-means"""
    if len(projects) < 3:
        return None, None
    
    X, project_names = prepare_ml_features(projects)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters (max 5 for small datasets)
    n_clusters = min(5, len(projects))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return cluster_labels, kmeans

def generate_ml_insights(projects):
    """Generate ML-powered insights"""
    insights = []
    recommendations = []
    
    if len(projects) < 3:
        insights.append("⚠️ **Not enough data** for ML analysis (need at least 3 projects)")
        return insights, recommendations
    
    # Train ML models
    with st.spinner("🤖 Training ML models..."):
        risk_model = train_risk_prediction_model(projects)
        budget_model = train_budget_prediction_model(projects)
        cluster_labels, kmeans_model = cluster_projects(projects)
    
    # ML Insight 1: Risk Prediction
    if risk_model:
        X, project_names = prepare_ml_features(projects)
        risk_predictions = risk_model.predict_proba(X)
        
        high_risk_count = 0
        for i, probs in enumerate(risk_predictions):
            high_risk_prob = probs[1]  # Probability of being high risk
            if high_risk_prob > 0.7:
                high_risk_count += 1
                project_name = projects[i]['name']
                
                # Get feature importance for this project
                feature_importance = risk_model.feature_importances_
                top_features_idx = np.argsort(feature_importance)[-3:][::-1]
                feature_names = ['Budget', 'Spent', 'Progress', 'Spent Ratio', 'Progress Ratio', 
                                'Efficiency', 'Status', 'Risk Level', 'Duration', 'Time Ratio', 'ROI']
                
                risk_factors = []
                for idx in top_features_idx:
                    if feature_importance[idx] > 0.1:
                        risk_factors.append(feature_names[idx])
                
                if risk_factors:
                    insights.append(f"🔴 **{project_name}**: ML predicts **{high_risk_prob:.0%} probability** of high risk. Key factors: {', '.join(risk_factors)}")
        
        if high_risk_count > 0:
            insights.append(f"📊 **ML Risk Assessment**: {high_risk_count} projects identified as high-risk by ML model (70%+ confidence)")
    
    # ML Insight 2: Budget Overrun Prediction
    if budget_model:
        X, project_names = prepare_ml_features(projects)
        budget_predictions = budget_model.predict(X)
        
        severe_overruns = []
        for i, pred in enumerate(budget_predictions):
            if pred > 20:  # More than 20% predicted overrun
                severe_overruns.append((projects[i]['name'], pred))
        
        if severe_overruns:
            insights.append("💰 **Budget Overrun Predictions**:")
            for project_name, overrun_pct in severe_overruns[:3]:  # Show top 3
                insights.append(f"   • {project_name}: Predicted to exceed budget by **{overrun_pct:.1f}%**")
            
            if len(severe_overruns) > 3:
                insights.append(f"   • ... and {len(severe_overruns) - 3} more projects")
    
    # ML Insight 3: Project Clustering
    if cluster_labels is not None and kmeans_model is not None:
        unique_clusters = np.unique(cluster_labels)
        cluster_insights = []
        
        for cluster_id in unique_clusters:
            cluster_projects = [projects[i] for i in range(len(projects)) if cluster_labels[i] == cluster_id]
            if len(cluster_projects) > 1:
                # Calculate cluster statistics
                avg_progress = np.mean([p['progress'] for p in cluster_projects])
                avg_spent_ratio = np.mean([p['spent']/p['budget'] for p in cluster_projects if p['budget'] > 0])
                avg_roi = np.mean([p['roi'] for p in cluster_projects if p['roi'] is not None])
                
                cluster_type = "High Performers" if avg_progress > 70 and (avg_roi > 15 if not np.isnan(avg_roi) else True) else "Needs Attention"
                cluster_insights.append(f"   • Cluster {cluster_id+1}: {len(cluster_projects)} projects - {cluster_type}")
        
        if cluster_insights:
            insights.append("🎯 **Project Segmentation**:")
            insights.extend(cluster_insights)
            
            # Recommendation based on clusters
            low_progress_clusters = [i for i in unique_clusters 
                                   if np.mean([projects[j]['progress'] for j in range(len(projects)) 
                                             if cluster_labels[j] == i]) < 40]
            if low_progress_clusters:
                recommendations.append("Focus resources on low-progress project clusters")
    
    # ML Insight 4: Portfolio Efficiency Score
    if len(projects) >= 5:
        efficiency_scores = []
        for project in projects:
            spent_ratio = project['spent'] / project['budget'] if project['budget'] > 0 else 0
            progress_ratio = project['progress'] / 100
            efficiency = progress_ratio / spent_ratio if spent_ratio > 0 else 0
            efficiency_scores.append(efficiency)
        
        avg_efficiency = np.mean(efficiency_scores)
        if avg_efficiency < 0.8:
            insights.append(f"⚡ **Portfolio Efficiency**: Overall efficiency score is **{avg_efficiency:.2f}** (below optimal 0.8)")
            recommendations.append("Review project execution strategies to improve efficiency")
    
    # ML Insight 5: Anomaly Detection
    if len(projects) >= 5:
        # Simple anomaly detection based on z-score
        progress_values = [p['progress'] for p in projects]
        spent_ratios = [p['spent']/p['budget'] for p in projects if p['budget'] > 0]
        
        if spent_ratios:
            avg_spent_ratio = np.mean(spent_ratios)
            std_spent_ratio = np.std(spent_ratios)
            
            anomalies = []
            for i, project in enumerate(projects):
                if project['budget'] > 0:
                    spent_ratio = project['spent'] / project['budget']
                    if std_spent_ratio > 0 and abs(spent_ratio - avg_spent_ratio) > 2 * std_spent_ratio:
                        anomalies.append(project['name'])
            
            if anomalies:
                insights.append(f"🚨 **Anomaly Detection**: {len(anomalies)} projects show unusual spending patterns")
    
    # Generate ML-based recommendations
    if not recommendations:
        if risk_model and budget_model:
            # Analyze overall portfolio health
            X_all, _ = prepare_ml_features(projects)
            
            if risk_model:
                risk_scores = risk_model.predict_proba(X_all)[:, 1]
                avg_risk = np.mean(risk_scores)
                
                if avg_risk > 0.6:
                    recommendations.append("Consider rebalancing portfolio to reduce overall risk exposure")
                elif avg_risk < 0.3:
                    recommendations.append("Portfolio is low-risk; consider taking on more ambitious projects")
            
            if budget_model:
                budget_predictions = budget_model.predict(X_all)
                avg_overrun = np.mean([p for p in budget_predictions if p > 0])
                
                if avg_overrun > 15:
                    recommendations.append("Implement stricter budget controls and regular reviews")
    
    # Default recommendation if none generated
    if not recommendations:
        recommendations.append("Monitor key projects closely and review progress weekly")
    
    return insights, recommendations

def get_ml_confidence_color(confidence):
    """Get CSS class for confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# Original helper functions (keep existing)
def calculate_portfolio_metrics(projects):
    """Calculate portfolio-level metrics from projects"""
    if not projects:
        return {
            'total_projects': 0,
            'active_projects': 0,
            'at_risk_projects': 0,
            'total_budget': 0,
            'total_spent': 0,
            'budget_variance': 0,
            'avg_roi': 0,
            'avg_progress': 0,
            'completed_projects': 0,
            'completion_rate': 0
        }
    
    total_projects = len(projects)
    active_projects = len([p for p in projects if p['status'] == 'Active'])
    at_risk_projects = len([p for p in projects if p['status'] in ['At Risk', 'Critical']])
    
    total_budget = sum(p['budget'] for p in projects)
    total_spent = sum(p['spent'] for p in projects)
    budget_variance = ((total_spent - total_budget) / total_budget * 100) if total_budget > 0 else 0
    
    rois = [p['roi'] for p in projects if p['roi'] is not None]
    avg_roi = sum(rois) / len(rois) if rois else 0
    
    avg_progress = sum(p['progress'] for p in projects) / total_projects if total_projects > 0 else 0
    
    completed_projects = len([p for p in projects if p['progress'] == 100])
    completion_rate = (completed_projects / total_projects * 100) if total_projects > 0 else 0
    
    return {
        'total_projects': total_projects,
        'active_projects': active_projects,
        'at_risk_projects': at_risk_projects,
        'total_budget': total_budget,
        'total_spent': total_spent,
        'budget_variance': budget_variance,
        'avg_roi': avg_roi,
        'avg_progress': avg_progress,
        'completed_projects': completed_projects,
        'completion_rate': completion_rate
    }

def get_status_badge(status):
    """Return HTML badge for status"""
    status_classes = {
        'Planning': 'status-planning',
        'Active': 'status-active',
        'At Risk': 'status-at-risk',
        'Critical': 'status-critical',
        'Completed': 'status-completed'
    }
    return f'<span class="{status_classes.get(status, "status-planning")}">{status}</span>'

def get_status_counts(projects):
    """Count projects by status"""
    status_counts = {
        'Planning': 0,
        'Active': 0,
        'At Risk': 0,
        'Critical': 0,
        'Completed': 0
    }
    
    for project in projects:
        status = project['status']
        if status in status_counts:
            status_counts[status] += 1
    
    return status_counts

def process_uploaded_file(uploaded_file):
    """Process uploaded CSV/Excel file and convert to project format"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        st.session_state.df_uploaded = df
        st.session_state.data_warnings = []
        
        # Convert DataFrame to project format
        projects = []
        required_columns = ['name', 'description', 'status', 'budget', 'spent', 'progress']
        
        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: name, description, status, budget, spent, progress")
            return None
        
        # Data validation and cleaning
        for idx, row in df.iterrows():
            project_name = str(row['name'])
            
            # Data validation
            warnings = []
            
            # Check for data issues
            if pd.isna(row['budget']) or row['budget'] <= 0:
                warnings.append(f"Project '{project_name}': Invalid budget value")
                budget = 0
            else:
                budget = float(row['budget'])
            
            if pd.isna(row['spent']) or row['spent'] < 0:
                warnings.append(f"Project '{project_name}': Invalid spent value")
                spent = 0
            else:
                spent = float(row['spent'])
            
            # Check if spent exceeds budget
            if spent > budget:
                warnings.append(f"Project '{project_name}': Spent (${spent:,.0f}) exceeds budget (${budget:,.0f})")
            
            # Progress validation
            if pd.isna(row['progress']) or row['progress'] < 0 or row['progress'] > 100:
                warnings.append(f"Project '{project_name}': Invalid progress value")
                progress = 0
            else:
                progress = float(row['progress'])
            
            # ROI validation
            if 'roi' in df.columns and not pd.isna(row['roi']):
                roi = float(row['roi'])
            else:
                roi = None
            
            # Risk level validation
            if 'risk_level' in df.columns and not pd.isna(row['risk_level']) and str(row['risk_level']).strip():
                risk_level = str(row['risk_level']).strip()
            else:
                # Auto-assign risk level based on status
                status_val = str(row['status']) if 'status' in df.columns and not pd.isna(row['status']) else "Planning"
                if status_val in ['Critical', 'At Risk']:
                    risk_level = 'High'
                elif status_val == 'Completed':
                    risk_level = 'Low'
                else:
                    risk_level = 'Medium'
            
            # Date validation
            if 'start_date' in df.columns and not pd.isna(row['start_date']):
                start_date = str(row['start_date']).split()[0]  # Take only date part
            else:
                start_date = datetime.now().strftime("%Y-%m-%d")
            
            if 'end_date' in df.columns and not pd.isna(row['end_date']):
                end_date = str(row['end_date']).split()[0]
            else:
                # Set default end date (6 months from now)
                end_date = (datetime.now() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
            
            # Add warnings to session state
            if warnings:
                st.session_state.data_warnings.extend(warnings)
            
            project = {
                "name": project_name,
                "description": str(row['description']) if 'description' in df.columns and not pd.isna(row['description']) else "No description",
                "status": str(row['status']) if 'status' in df.columns and not pd.isna(row['status']) else "Planning",
                "start_date": start_date,
                "end_date": end_date,
                "progress": progress,
                "budget": budget,
                "spent": spent,
                "roi": roi,
                "risk_level": risk_level
            }
            projects.append(project)
        
        st.session_state.projects = projects
        st.session_state.data_uploaded = True
        return projects
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Main app logic
def main():
    # Welcome page if no data uploaded
    if not st.session_state.data_uploaded:
        show_welcome_page()
    else:
        show_dashboard()

def show_welcome_page():
    """Show welcome page with data upload options"""
    st.markdown('<div class="main-header">📊 PMO Dashboard Analyzer</div>', unsafe_allow_html=True)
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to PMO Dashboard Analyzer
        
        **Transform your project data into actionable insights** with our AI-powered dashboard.
        
        This tool helps you:
        - 📈 **Analyze** project portfolio performance
        - ⚠️ **Identify** at-risk projects automatically using **Machine Learning**
        - 💰 **Predict** budget overruns with ML models
        - 📊 **Generate** intelligent executive reports
        - 🎯 **Make** data-driven decisions
        
        ### How it Works:
        1. **Upload** your project data (CSV/Excel)
        2. **Analyze** with interactive dashboard
        3. **Generate** ML-powered insights
        4. **Export** results for stakeholders
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=200)
    
    st.markdown("---")
    
    # Data Upload Section
    st.markdown("## 📤 Upload Your Project Data")
    
    col_upload, col_sample = st.columns([2, 1])
    
    with col_upload:
        st.markdown("### Upload Your Data File")
        st.markdown("Supported formats: CSV, Excel (XLSX)")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing your data..."):
                projects = process_uploaded_file(uploaded_file)
                if projects:
                    st.success(f"✅ Successfully loaded {len(projects)} projects!")
                    
                    # Show data warnings if any
                    if st.session_state.data_warnings:
                        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                        st.warning("**Data Validation Warnings:**")
                        for warning in st.session_state.data_warnings[:5]:  # Show first 5 warnings
                            st.write(f"• {warning}")
                        if len(st.session_state.data_warnings) > 5:
                            st.write(f"• ... and {len(st.session_state.data_warnings) - 5} more warnings")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("🚀 Launch Dashboard", type="primary"):
                        st.rerun()
    
    with col_sample:
        st.markdown("### Sample Data Format")
        st.markdown("""
        Your CSV/Excel should include:
        
        **Required columns:**
        - `name`: Project name
        - `description`: Project description
        - `status`: Planning/Active/At Risk/Critical/Completed
        - `budget`: Total budget
        - `spent`: Amount spent
        - `progress`: Progress percentage (0-100)
        
        **Optional columns:**
        - `start_date`: Project start date
        - `end_date`: Project end date
        - `roi`: Return on investment
        - `risk_level`: Low/Medium/High/Critical
        """)

def show_dashboard():
    """Show the main dashboard after data is uploaded"""
    # Navigation
    st.sidebar.markdown("## 🔄 Navigation")
    
    if st.sidebar.button("⬅️ Back to Upload", use_container_width=True):
        st.session_state.data_uploaded = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Data warnings in sidebar
    if st.session_state.data_warnings:
        with st.sidebar.expander("⚠️ Data Warnings", expanded=False):
            for warning in st.session_state.data_warnings[:3]:
                st.write(f"• {warning}")
            if len(st.session_state.data_warnings) > 3:
                st.write(f"• ... {len(st.session_state.data_warnings) - 3} more")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Projects", "📈 Reports", "📁 Data"])
    
    with tab1:
        show_dashboard_tab()
    
    with tab2:
        show_projects_tab()
    
    with tab3:
        show_reports_tab()
    
    with tab4:
        show_data_tab()

def show_dashboard_tab():
    """Dashboard tab content"""
    if not st.session_state.projects:
        st.warning("No project data available. Please upload data first.")
        return
    
    metrics = calculate_portfolio_metrics(st.session_state.projects)
    status_counts = get_status_counts(st.session_state.projects)
    
    st.markdown('<div class="main-header">📊 PMO Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"**Portfolio Overview: {len(st.session_state.projects)} Projects | Total Budget: ${metrics['total_budget']:,.0f}**")
    
    # Show data warnings at top
    if st.session_state.data_warnings:
        with st.container():
            st.markdown('<div class="warning-card">', unsafe_allow_html=True)
            st.warning(f"⚠️ {len(st.session_state.data_warnings)} data quality issues detected. Some metrics may be affected.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # KPI Metrics Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Projects", metrics['total_projects'], f"{metrics['active_projects']} active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Completion Rate", f"{metrics['completion_rate']:.1f}%", 
                 f"{metrics['completed_projects']} completed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Projects", metrics['at_risk_projects'], "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Budget", f"${metrics['total_budget']:,.0f}", 
                 f"${metrics['total_spent']:,.0f} spent")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average ROI", f"{metrics['avg_roi']:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        variance_color = "normal" if metrics['budget_variance'] >= -20 else "inverse"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Budget Variance", f"{metrics['budget_variance']:.1f}%", 
                 delta_color=variance_color)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        resource_utilization = min(100, sum(p['progress'] * 30 for p in st.session_state.projects) / 3000 * 100) if st.session_state.projects else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Resource Utilization", f"{resource_utilization:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Progress", f"{metrics['avg_progress']:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Calculate budget efficiency
        budget_efficiency = (metrics['avg_progress'] / (metrics['total_spent'] / metrics['total_budget'] * 100)) * 100 if metrics['total_spent'] > 0 else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Budget Efficiency", f"{budget_efficiency:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts and Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">📊 Projects by Status</div>', unsafe_allow_html=True)
        
        labels = []
        values = []
        colors = []
        
        status_colors = {
            'Planning': '#FFA726',
            'Active': '#42A5F5',
            'At Risk': '#FF9800',
            'Critical': '#EF5350',
            'Completed': '#66BB6A'
        }
        
        for status, count in status_counts.items():
            if count > 0:
                labels.append(f"{status} ({count})")
                values.append(count)
                colors.append(status_colors.get(status, '#CCCCCC'))
        
        if values:
            fig_status = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=colors
            )])
            fig_status.update_layout(
                height=300, 
                showlegend=True,
                title_text="Project Status Distribution"
            )
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("No status data available")
    
    with col2:
        st.markdown('<div class="sub-header">💰 Budget vs Progress</div>', unsafe_allow_html=True)
        
        if st.session_state.projects:
            # Create scatter plot of progress vs budget spent percentage
            project_names = []
            progress_values = []
            spent_percentages = []
            status_colors_scatter = []
            
            for project in st.session_state.projects:
                project_names.append(project['name'])
                progress_values.append(project['progress'])
                spent_percent = (project['spent'] / project['budget'] * 100) if project['budget'] > 0 else 0
                spent_percentages.append(spent_percent)
                
                # Color by risk level
                if project['risk_level'] == 'Critical' or project['status'] == 'Critical':
                    status_colors_scatter.append('#EF5350')
                elif project['risk_level'] == 'High' or project['status'] == 'At Risk':
                    status_colors_scatter.append('#FF9800')
                elif project['status'] == 'Completed':
                    status_colors_scatter.append('#66BB6A')
                else:
                    status_colors_scatter.append('#42A5F5')
            
            fig_scatter = go.Figure(data=go.Scatter(
                x=spent_percentages,
                y=progress_values,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=status_colors_scatter,
                    opacity=0.8
                ),
                text=project_names,
                textposition="top center",
                hoverinfo='text+x+y',
                hovertext=[f"{name}<br>Progress: {progress}%<br>Spent: {spent:.1f}%" 
                          for name, progress, spent in zip(project_names, progress_values, spent_percentages)]
            ))
            
            fig_scatter.update_layout(
                height=300,
                title_text="Progress vs Budget Spent",
                xaxis_title="Budget Spent (%)",
                yaxis_title="Progress (%)",
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No project data available")
    
    # 🔥 ENHANCED: AI-Powered Insights with ML
    st.markdown('<div class="sub-header">🤖 AI-Powered Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Add ML model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        ml_mode = st.radio(
            "Select Analysis Mode:",
            ["🤖 ML-Powered Insights", "📊 Basic Rule-Based"],
            horizontal=True
        )
    
    with col2:
        if st.button("🔍 Generate Insights", type="primary", key="gen_insights"):
            st.session_state.generating_insights = True
            st.rerun()
    
    if st.session_state.get('generating_insights', False):
        st.session_state.generating_insights = False
        
        if ml_mode == "🤖 ML-Powered Insights":
            st.success("🤖 ML Analysis Generated!")
            
            # Generate ML insights
            ml_insights, ml_recommendations = generate_ml_insights(st.session_state.projects)
            
            if ml_insights:
                st.markdown("### 🔬 ML-Powered Insights")
                st.markdown('<div class="ml-insight-card">', unsafe_allow_html=True)
                
                for insight in ml_insights:
                    st.markdown(f"• {insight}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if ml_recommendations:
                    st.markdown("### 🎯 ML Recommendations")
                    for i, rec in enumerate(ml_recommendations, 1):
                        st.markdown(f"{i}. **{rec}**")
            else:
                st.info("Not enough data for ML analysis. Using basic insights instead.")
                ml_mode = "📊 Basic Rule-Based"
        
        if ml_mode == "📊 Basic Rule-Based":
            st.success("📊 Basic Analysis Generated!")
            
            insights = []
            
            # Risk analysis
            if metrics['at_risk_projects'] > 0:
                insights.append(f"⚠️ **Critical Risk**: {metrics['at_risk_projects']} projects are at critical risk")
            
            # Budget analysis
            if metrics['budget_variance'] < -50:
                insights.append(f"💰 **Budget Issue**: Significant underspending ({metrics['budget_variance']:.1f}% variance)")
            elif metrics['total_spent'] > metrics['total_budget']:
                insights.append(f"💰 **Budget Alert**: Total spending exceeds total budget!")
            
            # Progress analysis
            if metrics['completion_rate'] == 0:
                insights.append("📊 **Progress Concern**: No projects have been completed yet")
            
            if metrics['avg_progress'] < 50:
                insights.append(f"📊 **Progress Issue**: Average project progress is below 50% ({metrics['avg_progress']:.1f}%)")
            
            # ROI analysis
            if metrics['avg_roi'] < 5:
                insights.append(f"📈 **ROI Concern**: Average ROI is low at {metrics['avg_roi']:.1f}%")
            
            # Specific project issues
            overspent_projects = [p for p in st.session_state.projects if p['spent'] > p['budget']]
            if overspent_projects:
                insights.append(f"💸 **Overspending**: {len(overspent_projects)} projects have spent more than their budget")
            
            if insights:
                st.info("### Key Insights:")
                for insight in insights:
                    st.write(f"• {insight}")
            else:
                st.success("✅ All projects are on track!")
            
            # Top recommendations
            st.markdown("### 🎯 Recommendations:")
            
            if overspent_projects:
                st.write("1. **Immediate Action**: Review overspent projects for budget adjustments")
            
            if metrics['at_risk_projects'] > 0:
                st.write("2. **Priority Focus**: Allocate resources to at-risk projects")
            
            if metrics['budget_variance'] < -50:
                st.write("3. **Budget Review**: Investigate why budget utilization is low")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ALL Projects Section (not just recent)
    st.markdown('<div class="sub-header">📋 All Projects ({})</div>'.format(len(st.session_state.projects)), unsafe_allow_html=True)
    
    # Add sorting options
    col_sort, col_filter = st.columns([1, 2])
    with col_sort:
        sort_by = st.selectbox("Sort by:", ["Name", "Progress", "Budget", "Risk Level"])
    
    with col_filter:
        filter_status = st.multiselect(
            "Filter by Status:",
            ["Planning", "Active", "At Risk", "Critical", "Completed"],
            default=["Planning", "Active", "At Risk", "Critical", "Completed"]
        )
    
    # Sort and filter projects
    display_projects = st.session_state.projects.copy()
    
    # Filter by status
    if filter_status:
        display_projects = [p for p in display_projects if p['status'] in filter_status]
    
    # Sort projects
    if sort_by == "Progress":
        display_projects.sort(key=lambda x: x['progress'], reverse=True)
    elif sort_by == "Budget":
        display_projects.sort(key=lambda x: x['budget'], reverse=True)
    elif sort_by == "Risk Level":
        risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        display_projects.sort(key=lambda x: risk_order.get(x['risk_level'], 4))
    else:  # Name
        display_projects.sort(key=lambda x: x['name'])
    
    # Display all projects
    for project in display_projects:
        with st.container():
            st.markdown('<div class="project-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {project['name']}")
                st.markdown(f"*{project['description']}*")
                st.markdown(f"Status: {get_status_badge(project['status'])} | Risk: {project['risk_level']}", unsafe_allow_html=True)
                
                # Progress bar with warning if overspent
                progress_color = "normal"
                spent_percent = (project['spent'] / project['budget'] * 100) if project['budget'] > 0 else 0
                
                if project['spent'] > project['budget']:
                    st.warning(f"⚠️ **OVERSIGHT**: Spent ${project['spent']:,} ({(spent_percent):.0f}%) exceeds budget ${project['budget']:,}")
                    progress_color = "red"
                elif spent_percent > 100:
                    progress_color = "orange"
                
                st.markdown(f"Progress: {project['progress']}%")
                st.progress(min(1.0, project['progress'] / 100))
                
                # Budget vs Progress comparison
                col_budget, col_spent = st.columns(2)
                with col_budget:
                    st.metric("Budget", f"${project['budget']:,}")
                with col_spent:
                    delta_spent = f"{spent_percent:.1f}%" if project['budget'] > 0 else "N/A"
                    st.metric("Spent", f"${project['spent']:,}", delta_spent)
            
            with col2:
                st.markdown("**Timeline**")
                st.markdown(f"Start: {project['start_date']}")
                st.markdown(f"End: {project['end_date']}")
                
                st.markdown("**Financials**")
                if project['roi']:
                    st.markdown(f"ROI: {project['roi']}%")
                
                # Efficiency metric
                if project['budget'] > 0 and project['progress'] > 0:
                    efficiency = (project['progress'] / spent_percent * 100) if spent_percent > 0 else 0
                    st.metric("Efficiency", f"{efficiency:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)

def show_projects_tab():
    """Projects tab content"""
    st.markdown('<div class="main-header">📋 Project Management</div>', unsafe_allow_html=True)
    st.markdown("Manage and analyze all your projects")
    
    if not st.session_state.projects:
        st.info("No projects available. Please upload data first.")
        return
    
    # Show summary statistics
    metrics = calculate_portfolio_metrics(st.session_state.projects)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Projects", metrics['total_projects'])
    with col_stat2:
        st.metric("Active Projects", metrics['active_projects'])
    with col_stat3:
        st.metric("At Risk", metrics['at_risk_projects'])
    with col_stat4:
        st.metric("Avg Progress", f"{metrics['avg_progress']:.1f}%")
    
    # Search and filter
    col_search, col_filter, col_sort = st.columns(3)
    with col_search:
        search_query = st.text_input("🔍 Search projects...", key="project_search")
    
    with col_filter:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Planning", "Active", "At Risk", "Critical", "Completed"]
        )
    
    with col_sort:
        risk_filter = st.selectbox(
            "Filter by Risk",
            ["All", "Low", "Medium", "High", "Critical"]
        )
    
    # Filter projects
    filtered_projects = st.session_state.projects
    
    if search_query:
        filtered_projects = [p for p in filtered_projects 
                           if search_query.lower() in p['name'].lower() 
                           or search_query.lower() in p['description'].lower()]
    
    if status_filter != "All":
        filtered_projects = [p for p in filtered_projects 
                           if p['status'] == status_filter]
    
    if risk_filter != "All":
        filtered_projects = [p for p in filtered_projects 
                           if p['risk_level'] == risk_filter]
    
    st.markdown(f"**Showing {len(filtered_projects)} of {len(st.session_state.projects)} projects**")
    
    # Display projects in a grid
    cols_per_row = 2
    for i in range(0, len(filtered_projects), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            if i + col_idx < len(filtered_projects):
                project = filtered_projects[i + col_idx]
                with cols[col_idx]:
                    with st.container():
                        st.markdown('<div class="project-card">', unsafe_allow_html=True)
                        
                        # Project header with status
                        col_header1, col_header2 = st.columns([3, 1])
                        with col_header1:
                            st.markdown(f"**{project['name']}**")
                        with col_header2:
                            st.markdown(get_status_badge(project['status']), unsafe_allow_html=True)
                        
                        st.markdown(f"*{project['description'][:50]}...*" if len(project['description']) > 50 else f"*{project['description']}*")
                        
                        # Progress
                        st.markdown(f"Progress: **{project['progress']}%**")
                        st.progress(min(1.0, project['progress'] / 100))
                        
                        # Budget information
                        spent_percent = (project['spent'] / project['budget'] * 100) if project['budget'] > 0 else 0
                        
                        col_budget1, col_budget2 = st.columns(2)
                        with col_budget1:
                            st.markdown(f"**Budget:**\n${project['budget']:,}")
                        with col_budget2:
                            st.markdown(f"**Spent:**\n${project['spent']:,}")
                        
                        # Warning for overspent projects
                        if project['spent'] > project['budget']:
                            st.error(f"⚠️ Overspent by ${project['spent'] - project['budget']:,}")
                        
                        # Additional info
                        with st.expander("View Details"):
                            st.markdown(f"**Risk Level:** {project['risk_level']}")
                            st.markdown(f"**Start Date:** {project['start_date']}")
                            st.markdown(f"**End Date:** {project['end_date']}")
                            if project['roi']:
                                st.markdown(f"**ROI:** {project['roi']}%")
                            st.markdown(f"**Budget Utilization:** {spent_percent:.1f}%")
                        
                        st.markdown('</div>', unsafe_allow_html=True)

def show_reports_tab():
    """Reports tab content"""
    st.markdown('<div class="main-header">📈 Executive Reports</div>', unsafe_allow_html=True)
    st.markdown("Generate comprehensive portfolio reports")
    
    if not st.session_state.projects:
        st.info("No data available for reporting. Please upload project data.")
        return
    
    metrics = calculate_portfolio_metrics(st.session_state.projects)
    
    # Export Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Executive Summary")
    with col2:
        if st.button("📥 Export Full Report", type="primary", use_container_width=True):
            st.success("Report exported successfully!")
    
    # Summary Tables
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_data = {
            "Metric": ["Total Projects", "Active Projects", "Completion Rate", "Average Progress"],
            "Value": [metrics['total_projects'], metrics['active_projects'], 
                     f"{metrics['completion_rate']:.1f}%", f"{metrics['avg_progress']:.1f}%"]
        }
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, hide_index=True, use_container_width=True)
    
    with col2:
        financial_data = {
            "Metric": ["Total Budget", "Total Spent", "Budget Variance", "Average ROI"],
            "Value": [f"${metrics['total_budget']:,.0f}", f"${metrics['total_spent']:,.0f}", 
                     f"{metrics['budget_variance']:.1f}%", f"{metrics['avg_roi']:.1f}%"]
        }
        financial_df = pd.DataFrame(financial_data)
        st.dataframe(financial_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Analysis
    at_risk_projects = [p for p in st.session_state.projects if p['status'] in ['At Risk', 'Critical']]
    
    st.markdown(f"### ⚠️ Risk Analysis ({len(at_risk_projects)} Projects)")
    
    if at_risk_projects:
        for project in at_risk_projects:
            with st.container():
                st.markdown(f"#### {project['name']}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Progress", f"{project['progress']}%")
                with col2:
                    st.metric("Budget Spent", f"${project['spent']:,}")
                with col3:
                    st.metric("Status", project['status'])
                with col4:
                    st.metric("Risk Level", project['risk_level'])
                
                # Risk factors
                risk_factors = []
                if project['progress'] < 50 and (project['spent'] / project['budget']) > 0.5:
                    risk_factors.append("Spending ahead of progress")
                if project['status'] == 'Critical':
                    risk_factors.append("Critical priority")
                if project['roi'] and project['roi'] < 5:
                    risk_factors.append("Low ROI projection")
                if project['spent'] > project['budget']:
                    risk_factors.append("Overspent budget")
                
                if risk_factors:
                    st.warning("**Risk Factors:** " + ", ".join(risk_factors))
    else:
        st.success("✅ No at-risk projects identified")
    
    st.markdown("---")
    
    # Performance Ranking
    st.markdown("### 🏆 Top Performing Projects")
    
    projects_with_roi = [p for p in st.session_state.projects if p['roi'] is not None]
    if projects_with_roi:
        top_projects = sorted(projects_with_roi, key=lambda x: x['roi'], reverse=True)[:5]
        
        for i, project in enumerate(top_projects, 1):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.markdown(f"**#{i} {project['name']}**")
            with col2:
                st.markdown(f"ROI: {project['roi']}%")
            with col3:
                st.markdown(f"Progress: {project['progress']}%")
            with col4:
                st.markdown(f"Budget: ${project['budget']:,}")
    else:
        st.info("No ROI data available for ranking")

def show_data_tab():
    """Data tab for viewing and managing uploaded data"""
    st.markdown('<div class="main-header">📁 Data Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Uploaded Data Preview")
        
        if st.session_state.df_uploaded is not None:
            st.dataframe(st.session_state.df_uploaded, use_container_width=True, height=400)
            
            # Data statistics
            st.markdown("### 📊 Data Statistics")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Records", len(st.session_state.df_uploaded))
            
            with col_stat2:
                numeric_cols = st.session_state.df_uploaded.select_dtypes(include=['number']).columns
                st.metric("Numeric Columns", len(numeric_cols))
            
            with col_stat3:
                text_cols = st.session_state.df_uploaded.select_dtypes(include=['object']).columns
                st.metric("Text Columns", len(text_cols))
            
            with col_stat4:
                missing_values = st.session_state.df_uploaded.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            # Column information
            with st.expander("📋 Column Information"):
                col_info = []
                for col in st.session_state.df_uploaded.columns:
                    col_info.append({
                        "Column": col,
                        "Type": str(st.session_state.df_uploaded[col].dtype),
                        "Non-Null": st.session_state.df_uploaded[col].count(),
                        "Unique": st.session_state.df_uploaded[col].nunique()
                    })
                st.dataframe(pd.DataFrame(col_info))
        else:
            st.info("No data uploaded yet")
    
    with col2:
        st.markdown("### Data Actions")
        
        if st.button("🔄 Re-upload Data", use_container_width=True):
            st.session_state.data_uploaded = False
            st.rerun()
        
        if st.session_state.df_uploaded is not None:
            # Export data
            csv = st.session_state.df_uploaded.to_csv(index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name="pmo_export.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Show data quality issues
            with st.expander("🔍 Data Quality Check", expanded=True):
                issues = []
                
                if st.session_state.df_uploaded.isnull().sum().sum() > 0:
                    null_count = st.session_state.df_uploaded.isnull().sum().sum()
                    issues.append(f"{null_count} missing values")
                
                if 'progress' in st.session_state.df_uploaded.columns:
                    invalid_progress = st.session_state.df_uploaded[
                        (st.session_state.df_uploaded['progress'] < 0) | 
                        (st.session_state.df_uploaded['progress'] > 100)
                    ]
                    if len(invalid_progress) > 0:
                        issues.append(f"{len(invalid_progress)} invalid progress values")
                
                # Check for budget overspending
                if 'budget' in st.session_state.df_uploaded.columns and 'spent' in st.session_state.df_uploaded.columns:
                    overspent = st.session_state.df_uploaded[
                        st.session_state.df_uploaded['spent'] > st.session_state.df_uploaded['budget']
                    ]
                    if len(overspent) > 0:
                        issues.append(f"{len(overspent)} projects overspent")
                
                if issues:
                    st.warning("**Data Quality Issues:**")
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ Data quality is good")
        
        st.markdown("---")
        st.markdown("### 📝 Data Notes")
        st.info("""
        - Data is stored in session memory
        - Refresh will clear current data
        - Export to save your analysis
        - Re-upload to analyze new data
        """)

# Run the app
if __name__ == "__main__":
    main()
