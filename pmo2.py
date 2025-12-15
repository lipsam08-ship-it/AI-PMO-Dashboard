import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import io

# 🔥 NEW ML IMPORTS
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

# 🔥 NEW ML HELPER FUNCTIONS (WITH ERROR HANDLING)

def prepare_ml_features(projects):
    """Prepare features for ML models with error handling"""
    features = []
    project_names = []
    
    for project in projects:
        try:
            # Feature engineering with safe defaults
            budget = float(project['budget']) if project['budget'] and project['budget'] > 0 else 100000
            spent = float(project['spent']) if project['spent'] and project['spent'] >= 0 else 0
            progress = float(project['progress']) if project['progress'] and 0 <= project['progress'] <= 100 else 0
            
            spent_ratio = spent / budget if budget > 0 else 0
            progress_ratio = progress / 100
            efficiency = progress_ratio / spent_ratio if spent_ratio > 0 else 0
            
            # Convert status to numerical
            status_map = {'Planning': 0, 'Active': 1, 'At Risk': 2, 'Critical': 3, 'Completed': 4}
            status_num = status_map.get(project['status'], 0)
            
            # Convert risk level to numerical
            risk_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
            risk_num = risk_map.get(project['risk_level'], 1)
            
            # Calculate days if dates are available
            try:
                start_date = datetime.strptime(str(project['start_date']), '%Y-%m-%d')
                end_date = datetime.strptime(str(project['end_date']), '%Y-%m-%d')
                days_duration = max(1, (end_date - start_date).days)
                days_elapsed = max(0, (datetime.now() - start_date).days)
                time_ratio = days_elapsed / days_duration if days_duration > 0 else 0.5
            except:
                days_duration = 180  # Default 6 months
                days_elapsed = 90
                time_ratio = 0.5
            
            # ROI with default
            roi = float(project['roi']) if project['roi'] is not None and not np.isnan(project['roi']) else 10
            
            # Create feature vector
            feature_vector = [
                budget,
                spent,
                progress,
                spent_ratio,
                progress_ratio,
                efficiency,
                status_num,
                risk_num,
                days_duration,
                time_ratio,
                roi
            ]
            
            features.append(feature_vector)
            project_names.append(project['name'])
            
        except Exception as e:
            # Skip problematic projects
            continue
    
    if not features:
        return np.array([]), []
    
    return np.array(features), project_names

def train_risk_prediction_model(projects):
    """Train ML model to predict project risk with error handling"""
    try:
        if len(projects) < 3:
            return None
        
        X, project_names = prepare_ml_features(projects)
        
        if len(X) < 3:
            return None
        
        # Create target labels based on current risk status
        y = []
        for project in projects:
            if project['status'] in ['Critical', 'At Risk'] or project['risk_level'] in ['High', 'Critical']:
                y.append(1)  # High risk
            else:
                y.append(0)  # Low/Medium risk
        
        y = np.array(y)
        
        if len(set(y)) < 2:  # Need both classes
            # If only one class, create synthetic second class
            y = np.zeros(len(projects))
            y[:min(1, len(y)-1)] = 1  # Mark first project as high risk
        
        # Train Random Forest classifier
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        
        return model
    except Exception as e:
        return None

def train_budget_prediction_model(projects):
    """Train ML model to predict final budget overrun with error handling"""
    try:
        if len(projects) < 3:
            return None
        
        X, project_names = prepare_ml_features(projects)
        
        if len(X) < 3:
            return None
        
        # Create target: predicted overspending percentage
        y = []
        for project in projects:
            spent = float(project['spent']) if project['spent'] and project['spent'] >= 0 else 0
            budget = float(project['budget']) if project['budget'] and project['budget'] > 0 else 100000
            progress = float(project['progress']) if project['progress'] and 0 <= project['progress'] <= 100 else 0
            
            spent_ratio = spent / budget if budget > 0 else 0
            progress_ratio = progress / 100 if progress > 0 else 0.01
            
            # Estimate final cost based on current burn rate
            estimated_final = spent / progress_ratio if progress_ratio > 0 else budget * 1.5
            overspend_pct = ((estimated_final - budget) / budget * 100) if budget > 0 else 0
            
            y.append(max(0, overspend_pct))  # Only positive overspending
        
        y = np.array(y)
        
        # Train Random Forest regressor
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        
        return model
    except Exception as e:
        return None

def cluster_projects(projects):
    """Cluster projects into similar groups using K-means with error handling"""
    try:
        if len(projects) < 3:
            return None, None
        
        X, project_names = prepare_ml_features(projects)
        
        if len(X) < 3:
            return None, None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters (max 4 for small datasets)
        n_clusters = min(4, max(2, len(projects) // 2))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        return cluster_labels, kmeans
    except Exception as e:
        return None, None

def generate_ml_insights(projects):
    """Generate ML-powered insights with robust error handling"""
    insights = []
    recommendations = []
    
    try:
        if len(projects) < 3:
            insights.append("⚠️ **Not enough data** for ML analysis (need at least 3 projects)")
            return insights, recommendations
        
        # Train ML models
        risk_model = train_risk_prediction_model(projects)
        budget_model = train_budget_prediction_model(projects)
        cluster_labels, kmeans_model = cluster_projects(projects)
        
        # ML Insight 1: Risk Prediction
        if risk_model is not None:
            X, project_names = prepare_ml_features(projects)
            
            if len(X) > 0:
                try:
                    risk_predictions = risk_model.predict_proba(X)
                    
                    high_risk_count = 0
                    for i, probs in enumerate(risk_predictions):
                        if len(probs) > 1:
                            high_risk_prob = probs[1]  # Probability of being high risk
                        else:
                            high_risk_prob = probs[0]
                        
                        if high_risk_prob > 0.7 and i < len(projects):
                            high_risk_count += 1
                            project_name = projects[i]['name']
                            
                            # Get top risk factors
                            if hasattr(risk_model, 'feature_importances_'):
                                feature_importance = risk_model.feature_importances_
                                if len(feature_importance) > 0:
                                    top_features_idx = np.argsort(feature_importance)[-3:][::-1]
                                    feature_names = ['Budget', 'Spent', 'Progress', 'Spent Ratio', 'Progress Ratio', 
                                                    'Efficiency', 'Status', 'Risk Level', 'Duration', 'Time Ratio', 'ROI']
                                    
                                    risk_factors = []
                                    for idx in top_features_idx:
                                        if idx < len(feature_names) and feature_importance[idx] > 0.1:
                                            risk_factors.append(feature_names[idx])
                                    
                                    if risk_factors:
                                        insights.append(f"🔴 **{project_name}**: ML predicts **{high_risk_prob:.0%} probability** of high risk")
                    
                    if high_risk_count > 0:
                        insights.append(f"📊 **ML Risk Assessment**: {high_risk_count} projects identified as high-risk by ML model")
                except:
                    pass
        
        # ML Insight 2: Budget Overrun Prediction
        if budget_model is not None:
            X, project_names = prepare_ml_features(projects)
            
            if len(X) > 0:
                try:
                    budget_predictions = budget_model.predict(X)
                    
                    severe_overruns = []
                    for i, pred in enumerate(budget_predictions):
                        if i < len(projects) and pred > 15:  # More than 15% predicted overrun
                            severe_overruns.append((projects[i]['name'], pred))
                    
                    if severe_overruns:
                        insights.append("💰 **Budget Overrun Predictions**:")
                        for project_name, overrun_pct in severe_overruns[:3]:  # Show top 3
                            insights.append(f"   • {project_name}: Predicted to exceed budget by **{overrun_pct:.1f}%**")
                except:
                    pass
        
        # ML Insight 3: Project Clustering
        if cluster_labels is not None and kmeans_model is not None:
            try:
                unique_clusters = np.unique(cluster_labels)
                
                cluster_insights = []
                for cluster_id in unique_clusters:
                    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_projects = [projects[i] for i in cluster_indices if i < len(projects)]
                    
                    if len(cluster_projects) > 1:
                        # Calculate cluster statistics
                        progress_values = [p['progress'] for p in cluster_projects if 'progress' in p]
                        roi_values = [p['roi'] for p in cluster_projects if p.get('roi') is not None and not np.isnan(p['roi'])]
                        
                        if progress_values:
                            avg_progress = np.mean(progress_values)
                            cluster_type = "High Performers" if avg_progress > 70 else "Needs Attention"
                            cluster_insights.append(f"   • Cluster {cluster_id+1}: {len(cluster_projects)} projects - {cluster_type}")
                
                if cluster_insights:
                    insights.append("🎯 **Project Segmentation**:")
                    insights.extend(cluster_insights)
            except:
                pass
        
        # ML Insight 4: Portfolio Efficiency
        if len(projects) >= 3:
            try:
                efficiency_scores = []
                for project in projects:
                    spent = float(project['spent']) if project['spent'] and project['spent'] >= 0 else 0
                    budget = float(project['budget']) if project['budget'] and project['budget'] > 0 else 100000
                    progress = float(project['progress']) if project['progress'] and 0 <= project['progress'] <= 100 else 0
                    
                    spent_ratio = spent / budget if budget > 0 else 0
                    progress_ratio = progress / 100
                    efficiency = progress_ratio / spent_ratio if spent_ratio > 0 else 0
                    efficiency_scores.append(efficiency)
                
                if efficiency_scores:
                    avg_efficiency = np.mean(efficiency_scores)
                    if avg_efficiency < 0.8:
                        insights.append(f"⚡ **Portfolio Efficiency**: Overall efficiency score is **{avg_efficiency:.2f}** (below optimal 0.8)")
                        recommendations.append("Review project execution strategies to improve efficiency")
            except:
                pass
        
        # Generate ML-based recommendations
        if not recommendations:
            recommendations.append("Monitor key projects closely and review progress weekly")
        
        # If no ML insights were generated, add a fallback
        if not insights:
            insights.append("📊 **Basic Portfolio Analysis**:")
            insights.append(f"   • Total Projects: {len(projects)}")
            
            at_risk_count = len([p for p in projects if p.get('status') in ['At Risk', 'Critical'] or p.get('risk_level') in ['High', 'Critical']])
            if at_risk_count > 0:
                insights.append(f"   • At-Risk Projects: {at_risk_count}")
            
            avg_progress = np.mean([p.get('progress', 0) for p in projects])
            insights.append(f"   • Average Progress: {avg_progress:.1f}%")
        
    except Exception as e:
        # Fallback to basic insights if ML fails
        insights = ["⚠️ **ML Analysis Temporarily Unavailable** - Showing basic insights"]
        insights.append(f"📊 **Portfolio Summary**: {len(projects)} projects loaded")
        recommendations = ["Please check your data format and try again"]
    
    return insights, recommendations

def get_ml_confidence_color(confidence):
    """Get CSS class for confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# ORIGINAL HELPER FUNCTIONS (keep existing)
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
    active_projects = len([p for p in projects if p.get('status') == 'Active'])
    at_risk_projects = len([p for p in projects if p.get('status') in ['At Risk', 'Critical']])
    
    total_budget = sum(p.get('budget', 0) for p in projects)
    total_spent = sum(p.get('spent', 0) for p in projects)
    budget_variance = ((total_spent - total_budget) / total_budget * 100) if total_budget > 0 else 0
    
    rois = [p.get('roi') for p in projects if p.get('roi') is not None and not np.isnan(p.get('roi'))]
    avg_roi = sum(rois) / len(rois) if rois else 0
    
    avg_progress = sum(p.get('progress', 0) for p in projects) / total_projects if total_projects > 0 else 0
    
    completed_projects = len([p for p in projects if p.get('progress', 0) == 100])
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
    return f'<span class="{status_classes.get(status, 'status-planning')}">{status}</span>'

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
        status = project.get('status', 'Planning')
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
            try:
                project_name = str(row['name']) if pd.notna(row['name']) else f"Project_{idx+1}"
                
                # Data validation
                warnings = []
                
                # Check for data issues
                budget = float(row['budget']) if pd.notna(row['budget']) and row['budget'] > 0 else 100000
                if pd.isna(row['budget']) or row['budget'] <= 0:
                    warnings.append(f"Project '{project_name}': Invalid budget value, using default")
                
                spent = float(row['spent']) if pd.notna(row['spent']) and row['spent'] >= 0 else 0
                if pd.isna(row['spent']) or row['spent'] < 0:
                    warnings.append(f"Project '{project_name}': Invalid spent value, using 0")
                
                # Check if spent exceeds budget
                if spent > budget:
                    warnings.append(f"Project '{project_name}': Spent (${spent:,.0f}) exceeds budget (${budget:,.0f})")
                
                # Progress validation
                progress = float(row['progress']) if pd.notna(row['progress']) and 0 <= row['progress'] <= 100 else 0
                if pd.isna(row['progress']) or row['progress'] < 0 or row['progress'] > 100:
                    warnings.append(f"Project '{project_name}': Invalid progress value, using 0")
                
                # ROI validation
                roi = float(row['roi']) if 'roi' in df.columns and pd.notna(row['roi']) else None
                
                # Risk level validation
                if 'risk_level' in df.columns and pd.notna(row['risk_level']) and str(row['risk_level']).strip():
                    risk_level = str(row['risk_level']).strip()
                else:
                    # Auto-assign risk level based on status
                    status_val = str(row['status']) if 'status' in df.columns and pd.notna(row['status']) else "Planning"
                    if status_val in ['Critical', 'At Risk']:
                        risk_level = 'High'
                    elif status_val == 'Completed':
                        risk_level = 'Low'
                    else:
                        risk_level = 'Medium'
                
                # Date validation
                if 'start_date' in df.columns and pd.notna(row['start_date']):
                    try:
                        start_date = str(row['start_date']).split()[0]
                    except:
                        start_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    start_date = datetime.now().strftime("%Y-%m-%d")
                
                if 'end_date' in df.columns and pd.notna(row['end_date']):
                    try:
                        end_date = str(row['end_date']).split()[0]
                    except:
                        end_date = (datetime.now() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
                else:
                    # Set default end date (6 months from now)
                    end_date = (datetime.now() + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
                
                # Add warnings to session state
                if warnings:
                    st.session_state.data_warnings.extend(warnings)
                
                project = {
                    "name": project_name,
                    "description": str(row['description']) if 'description' in df.columns and pd.notna(row['description']) else "No description",
                    "status": str(row['status']) if 'status' in df.columns and pd.notna(row['status']) else "Planning",
                    "start_date": start_date,
                    "end_date": end_date,
                    "progress": progress,
                    "budget": budget,
                    "spent": spent,
                    "roi": roi,
                    "risk_level": risk_level
                }
                projects.append(project)
                
            except Exception as e:
                st.warning(f"Skipping row {idx+1}: {str(e)}")
                continue
        
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
        resource_utilization = min(100, sum(p.get('progress', 0) * 30 for p in st.session_state.projects) / 3000 * 100) if st.session_state.projects else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Resource Utilization", f"{resource_utilization:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Progress", f"{metrics['avg_progress']:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Calculate budget efficiency
        budget_efficiency = (metrics['avg_progress'] / (metrics['total_spent'] / metrics['total_budget'] * 100)) * 100 if metrics['total_spent'] > 0 and metrics['total_budget'] > 0 else 0
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
                project_names.append(project.get('name', 'Unknown'))
                progress_values.append(project.get('progress', 0))
                spent_percent = (project.get('spent', 0) / project.get('budget', 1) * 100) if project.get('budget', 0) > 0 else 0
                spent_percentages.append(spent_percent)
                
                # Color by risk level
                if project.get('risk_level') == 'Critical' or project.get('status') == 'Critical':
                    status_colors_scatter.append('#EF5350')
                elif project.get('risk_level') == 'High' or project.get('status') == 'At Risk':
                    status_colors_scatter.append('#FF9800')
                elif project.get('status') == 'Completed':
                    status_colors_scatter.append('#66BB6A')
                else:
                    status_colors_scatter.append('#42A5F5')
            
            fig_scatter = go.Figure(data=go.Scatter(
                x=spent_percentages,
                y=progress_values,
                mode='markers',
                marker=dict(
                    size=15,
                    color=status_colors_scatter,
                    opacity=0.8
                ),
                text=project_names,
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
        generate_button = st.button("🔍 Generate Insights", type="primary", key="gen_insights")
    
    if generate_button:
        with st.spinner("Generating insights..."):
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
            
            else:  # Basic Rule-Based
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
                overspent_projects = [p for p in st.session_state.projects if p.get('spent', 0) > p.get('budget', 0)]
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
        display_projects = [p for p in display_projects if p.get('status') in filter_status]
    
    # Sort projects
    if sort_by == "Progress":
        display_projects.sort(key=lambda x: x.get('progress', 0), reverse=True)
    elif sort_by == "Budget":
        display_projects.sort(key=lambda x: x.get('budget', 0), reverse=True)
    elif sort_by == "Risk Level":
        risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        display_projects.sort(key=lambda x: risk_order.get(x.get('risk_level', 'Medium'), 4))
    else:  # Name
        display_projects.sort(key=lambda x: x.get('name', ''))
    
    # Display all projects
    for project in display_projects:
        with st.container():
            st.markdown('<div class="project-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {project.get('name', 'Unnamed Project')}")
                st.markdown(f"*{project.get('description', 'No description')}*")
                st.markdown(f"Status: {get_status_badge(project.get('status', 'Planning'))} | Risk: {project.get('risk_level', 'Medium')}", unsafe_allow_html=True)
                
                # Progress bar with warning if overspent
                spent_percent = (project.get('spent', 0) / project.get('budget', 1) * 100) if project.get('budget', 0) > 0 else 0
                
                if project.get('spent', 0) > project.get('budget', 0):
                    st.warning(f"⚠️ **OVERSIGHT**: Spent ${project.get('spent', 0):,} ({(spent_percent):.0f}%) exceeds budget ${project.get('budget', 0):,}")
                
                st.markdown(f"Progress: {project.get('progress', 0)}%")
                st.progress(min(1.0, project.get('progress', 0) / 100))
                
                # Budget vs Progress comparison
                col_budget, col_spent = st.columns(2)
                with col_budget:
                    st.metric("Budget", f"${project.get('budget', 0):,}")
                with col_spent:
                    delta_spent = f"{spent_percent:.1f}%" if project.get('budget', 0) > 0 else "N/A"
                    st.metric("Spent", f"${project.get('spent', 0):,}", delta_spent)
            
            with col2:
                st.markdown("**Timeline**")
                st.markdown(f"Start: {project.get('start_date', 'N/A')}")
                st.markdown(f"End: {project.get('end_date', 'N/A')}")
                
                st.markdown("**Financials**")
                if project.get('roi') is not None:
                    st.markdown(f"ROI: {project.get('roi')}%")
                
                # Efficiency metric
                if project.get('budget', 0) > 0 and project.get('progress', 0) > 0:
                    efficiency = (project.get('progress', 0) / spent_percent * 100) if spent_percent > 0 else 0
                    st.metric("Efficiency", f"{efficiency:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ... (keep the rest of the functions as they were - show_projects_tab, show_reports_tab, show_data_tab)

# IMPORTANT: Copy the show_projects_tab(), show_reports_tab(), and show_data_tab() functions 
# from your original code here - they should work as is since I didn't modify them

# Run the app
if __name__ == "__main__":
    main()
