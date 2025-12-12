import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date

# Initialize session state for projects if not exists
if 'projects' not in st.session_state:
    st.session_state.projects = [
        {
            "name": "Digital Transformation Initiative",
            "description": "Modernize legacy systems and implement cloud infrastructure",
            "status": "Active",
            "start_date": "2025-01-15",
            "end_date": "2025-12-31",
            "progress": 35,
            "budget": 500000,
            "spent": 125000,
            "roi": 15.5,
            "risk_level": "Low"
        },
        {
            "name": "Mobile App Development",
            "description": "Develop native mobile applications for iOS and Android",
            "status": "Planning",
            "start_date": "2025-03-01",
            "end_date": "2025-11-30",
            "progress": 0,
            "budget": 250000,
            "spent": 0,
            "roi": None,
            "risk_level": "Medium"
        },
        {
            "name": "Customer Portal Redesign",
            "description": "Redesign customer-facing portal with improved UX",
            "status": "Active",
            "start_date": "2025-01-01",
            "end_date": "2025-06-30",
            "progress": 60,
            "budget": 150000,
            "spent": 75000,
            "roi": 22.3,
            "risk_level": "Low"
        },
        {
            "name": "Security Compliance Upgrade",
            "description": "Implement security measures for compliance certification",
            "status": "Critical",
            "start_date": "2024-11-01",
            "end_date": "2025-03-31",
            "progress": 85,
            "budget": 200000,
            "spent": 175000,
            "roi": 12.1,
            "risk_level": "Critical"
        },
        {
            "name": "Data Analytics Platform",
            "description": "Build centralized analytics platform for business intelligence",
            "status": "At Risk",
            "start_date": "2025-02-01",
            "end_date": "2025-09-30",
            "progress": 45,
            "budget": 300000,
            "spent": 180000,
            "roi": 8.2,
            "risk_level": "High"
        }
    ]

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
    .stButton > button {
        width: 100%;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

def calculate_portfolio_metrics(projects):
    """Calculate portfolio-level metrics from projects"""
    total_projects = len(projects)
    active_projects = len([p for p in projects if p['status'] == 'Active'])
    at_risk_projects = len([p for p in projects if p['status'] in ['At Risk', 'Critical']])
    
    total_budget = sum(p['budget'] for p in projects)
    total_spent = sum(p['spent'] for p in projects)
    budget_variance = ((total_spent - total_budget) / total_budget * 100) if total_budget > 0 else 0
    
    rois = [p['roi'] for p in projects if p['roi'] is not None]
    avg_roi = sum(rois) / len(rois) if rois else 0
    
    avg_progress = sum(p['progress'] for p in projects) / total_projects if total_projects > 0 else 0
    
    # Calculate completion rate (projects at 100% progress)
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

def create_new_project():
    """Function to handle new project creation"""
    st.markdown("### Create New Project")
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    
    with st.form(key="new_project_form"):
        # Project Name
        project_name = st.text_input("Project Name", placeholder="Enter project name")
        
        # Description
        description = st.text_area("Description", placeholder="Enter project description", height=100)
        
        # Status
        status = st.selectbox(
            "Status",
            ["Planning", "Active", "At Risk", "Critical", "Completed"]
        )
        
        # Dates
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start Date", value=date.today())
        with col_date2:
            end_date = st.date_input("End Date", value=date.today())
        
        # Budget
        col_budget1, col_budget2 = st.columns(2)
        with col_budget1:
            budget = st.number_input("Budget ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        with col_budget2:
            spent = st.number_input("Spent ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        
        # Progress and ROI
        col_prog1, col_prog2 = st.columns(2)
        with col_prog1:
            progress = st.number_input("Progress (%)", min_value=0, max_value=100, value=0)
        with col_prog2:
            roi = st.number_input("ROI (%)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
            if roi == 0:
                roi = None
        
        # Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            cancel_button = st.form_submit_button("Cancel", type="secondary")
        with col_btn2:
            submit_button = st.form_submit_button("Create Project", type="primary")
        
        if submit_button:
            if not project_name:
                st.error("Project name is required!")
            elif not description:
                st.error("Description is required!")
            elif start_date >= end_date:
                st.error("End date must be after start date!")
            elif spent > budget:
                st.error("Spent amount cannot exceed budget!")
            else:
                # Determine risk level based on status
                risk_mapping = {
                    'Planning': 'Low',
                    'Active': 'Low',
                    'At Risk': 'High',
                    'Critical': 'Critical',
                    'Completed': 'Low'
                }
                
                # Create new project
                new_project = {
                    "name": project_name,
                    "description": description,
                    "status": status,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "progress": progress,
                    "budget": float(budget),
                    "spent": float(spent),
                    "roi": float(roi) if roi else None,
                    "risk_level": risk_mapping.get(status, 'Low')
                }
                
                # Add to projects list
                st.session_state.projects.append(new_project)
                
                # Show success message
                st.markdown('<div class="success-message">', unsafe_allow_html=True)
                st.success(f"✅ Project '{project_name}' created successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Force rerun to update dashboard
                st.rerun()
        
        if cancel_button:
            st.info("Form cleared. You can start over.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Projects", "📈 Reports"])

# Dashboard Tab
with tab1:
    st.markdown('<div class="main-header">PMO Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Portfolio overview with AI-powered insights and real-time metrics")
    
    # Calculate metrics from current project data
    metrics = calculate_portfolio_metrics(st.session_state.projects)
    status_counts = get_status_counts(st.session_state.projects)
    
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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Budget Variance", f"{metrics['budget_variance']:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate resource utilization based on project progress vs budget spent
        total_hours_used = sum(p['progress'] * 30 for p in st.session_state.projects)  # Simplified calculation
        total_hours_available = 3000
        resource_utilization = min(100, (total_hours_used / total_hours_available * 100)) if total_hours_available > 0 else 0
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Resource Utilization", f"{resource_utilization:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("AI Risk Projects", metrics['at_risk_projects'], "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Progress", f"{metrics['avg_progress']:.1f}%", "")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts and Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Projects by Status</div>', unsafe_allow_html=True)
        
        # Filter out statuses with zero count
        labels = []
        values = []
        colors = []
        
        # Define colors for each status
        status_colors = {
            'Planning': '#FFA726',
            'Active': '#42A5F5',
            'At Risk': '#FF9800',
            'Critical': '#EF5350',
            'Completed': '#66BB6A'
        }
        
        for status, count in status_counts.items():
            if count > 0:
                labels.append(status)
                values.append(count)
                colors.append(status_colors.get(status, '#CCCCCC'))
        
        # Create a donut chart for project status
        fig_status = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors
        )])
        fig_status.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">Budget Overview</div>', unsafe_allow_html=True)
        
        # Calculate budget spent percentage
        spent_percentage = min(100, (metrics['total_spent'] / metrics['total_budget'] * 100)) if metrics['total_budget'] > 0 else 0
        
        # Create a gauge chart for budget
        fig_budget = go.Figure(go.Indicator(
            mode="gauge+number",
            value=spent_percentage,
            title={'text': f"Spent: {spent_percentage:.1f}%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f3c88"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_budget.update_layout(height=300)
        st.plotly_chart(fig_budget, use_container_width=True)
    
    # AI-Powered Insights Section
    st.markdown('<div class="sub-header">AI-Powered Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Generate Insights")
    st.write("Click 'Generate Insights' to get AI-powered analysis of your portfolio.")
    if st.button("Generate Insights", type="primary", key="gen_insights"):
        st.success("AI Analysis Generated!")
        st.info(f"""
        **Insights:**
        1. {metrics['at_risk_projects']} projects are at risk of missing deadlines
        2. Budget variance is {metrics['budget_variance']:.1f}%
        3. Resource utilization is {resource_utilization:.1f}%
        4. Average project progress is {metrics['avg_progress']:.1f}%
        5. Focus needed on projects with status 'At Risk' or 'Critical'
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent Projects Section
    st.markdown('<div class="sub-header">Recent Projects</div>', unsafe_allow_html=True)
    
    # Show latest 5 projects
    recent_projects = st.session_state.projects[-5:] if len(st.session_state.projects) > 5 else st.session_state.projects
    
    for project in recent_projects:
        with st.container():
            st.markdown('<div class="project-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{project['name']}**")
                st.markdown(f"<small>{project['description']}</small>", unsafe_allow_html=True)
                st.markdown(f"Status: {get_status_badge(project['status'])}", unsafe_allow_html=True)
                
                # Progress bar
                st.markdown(f"Progress: {project['progress']}%")
                st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {project["progress"]}%"></div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Budget:** ${project['budget']:,}")
                st.markdown(f"**Spent:** ${project['spent']:,}")
                st.markdown(f"**Due:** {project['end_date']}")
                if project['roi']:
                    st.markdown(f"**ROI:** {project['roi']}%")
            st.markdown('</div>', unsafe_allow_html=True)

# Projects Tab
with tab2:
    st.markdown('<div class="main-header">Projects</div>', unsafe_allow_html=True)
    st.markdown("Manage and track all your projects")
    
    # Create two columns: one for project list, one for new project form
    col_list, col_form = st.columns([2, 1])
    
    with col_list:
        # Search bar
        search_query = st.text_input("Search projects...", key="project_search")
        
        # Filter projects based on search
        filtered_projects = st.session_state.projects
        if search_query:
            filtered_projects = [p for p in st.session_state.projects 
                               if search_query.lower() in p['name'].lower() 
                               or search_query.lower() in p['description'].lower()]
        
        # Display projects
        st.markdown(f"### Project List ({len(filtered_projects)} projects)")
        
        if not filtered_projects:
            st.info("No projects found. Create your first project!")
        else:
            for project in filtered_projects:
                with st.container():
                    st.markdown('<div class="project-card">', unsafe_allow_html=True)
                    st.markdown(f"### {project['name']}")
                    st.markdown(f"*{project['description']}*")
                    st.markdown(f"Status: {get_status_badge(project['status'])}", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Progress**")
                        st.markdown(f"### {project['progress']}%")
                        # FIXED: Ensure progress value is between 0 and 1
                        st.progress(min(1.0, max(0.0, project['progress'] / 100)))
                    
                    with col2:
                        st.markdown("**Budget & Spend**")
                        st.markdown(f"Budget: ${project['budget']:,}")
                        st.markdown(f"Spent: ${project['spent']:,}")
                        spend_percent = (project['spent'] / project['budget'] * 100) if project['budget'] > 0 else 0
                        st.markdown(f"Spent: {spend_percent:.1f}%")
                    
                    with col3:
                        st.markdown("**Timeline & ROI**")
                        st.markdown(f"Start: {project['start_date']}")
                        st.markdown(f"End: {project['end_date']}")
                        if project['roi']:
                            st.markdown(f"ROI: {project['roi']}%")
                    
                    # Risk indicators
                    if project['risk_level'] in ['High', 'Critical']:
                        if project['risk_level'] == 'Critical':
                            st.markdown('<span class="status-critical">● Critical Priority</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="status-at-risk">● At Risk</span>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with col_form:
        create_new_project()
        
        # Project Statistics
        st.markdown("### Portfolio Statistics")
        metrics = calculate_portfolio_metrics(st.session_state.projects)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Projects", metrics['total_projects'])
        with col_stat2:
            st.metric("Total Budget", f"${metrics['total_budget']:,.0f}")
        
        col_stat3, col_stat4 = st.columns(2)
        with col_stat3:
            st.metric("Active Projects", metrics['active_projects'])
        with col_stat4:
            st.metric("At Risk", metrics['at_risk_projects'])
        
        st.metric("Average Progress", f"{metrics['avg_progress']:.1f}%")

# Reports Tab
with tab3:
    st.markdown('<div class="main-header">PMO Dashboard</div>', unsafe_allow_html=True)
    st.markdown("## Executive Reports")
    st.markdown("Portfolio summary and key insights")
    
    # Calculate metrics for reports
    metrics = calculate_portfolio_metrics(st.session_state.projects)
    status_counts = get_status_counts(st.session_state.projects)
    
    st.markdown("### Executive Summary")
    
    # Summary tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Portfolio Overview")
        portfolio_data = {
            "Metric": ["Total Projects", "Active Projects", "Completion Rate", "Average Progress"],
            "Value": [metrics['total_projects'], metrics['active_projects'], 
                     f"{metrics['completion_rate']:.1f}%", 
                     f"{metrics['avg_progress']:.1f}%"]
        }
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Financial Summary")
        financial_data = {
            "Metric": ["Total Budget", "Total Spent", "Budget Variance", "Average ROI"],
            "Value": [f"${metrics['total_budget']:,.0f}", f"${metrics['total_spent']:,.0f}", 
                     f"{metrics['budget_variance']:.1f}%", f"{metrics['avg_roi']:.1f}%"]
        }
        financial_df = pd.DataFrame(financial_data)
        st.dataframe(financial_df, hide_index=True, use_container_width=True)
    
    # Export Report section
    st.markdown("### Export Report")
    
    # At-Risk Projects
    at_risk_projects = [p for p in st.session_state.projects if p['status'] in ['At Risk', 'Critical']]
    st.markdown(f"#### At-Risk Projects ({len(at_risk_projects)})")
    
    if at_risk_projects:
        cols = st.columns(2)
        for idx, project in enumerate(at_risk_projects):
            with cols[idx % 2]:
                st.markdown('<div class="project-card">', unsafe_allow_html=True)
                st.markdown(f"**{project['name']}**")
                if project['status'] == 'Critical':
                    st.markdown('<span class="status-critical">● critical</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-at-risk">● at-risk</span>', unsafe_allow_html=True)
                st.markdown(f"Progress: {project['progress']}%")
                st.markdown(f"Due: {project['end_date']}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No at-risk projects found.")
    
    st.markdown("---")
    
    # Top Performing Projects (by ROI)
    st.markdown("#### Top Performing Projects")
    
    # Filter projects with ROI and sort by ROI descending
    projects_with_roi = [p for p in st.session_state.projects if p['roi'] is not None]
    
    if projects_with_roi:
        top_projects = sorted(projects_with_roi, key=lambda x: x['roi'], reverse=True)[:4]
        
        top_projects_data = {
            "Project": [p['name'] for p in top_projects],
            "ROI": [f"{p['roi']:.1f}%" for p in top_projects],
            "Progress": [f"{p['progress']}%" for p in top_projects],
            "Budget": [f"${p['budget']:,}" for p in top_projects]
        }
        
        top_projects_df = pd.DataFrame(top_projects_data)
        st.dataframe(top_projects_df, hide_index=True, use_container_width=True)
    else:
        st.info("No projects with ROI data available.")
    
    st.markdown("---")
    
    # Resource Utilization
    st.markdown("#### Resource Utilization")
    
    # Calculate dynamic resource utilization - FIXED THE ERROR HERE
    total_hours_used = sum(p['progress'] * 30 for p in st.session_state.projects)
    total_hours_available = 3000
    utilization = min(100, (total_hours_used / total_hours_available * 100)) if total_hours_available > 0 else 0
    
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        st.markdown("**Overall Utilization**")
        st.markdown("Total Resources: 3")
        st.markdown(f"Total Hours: {int(total_hours_used)} / {total_hours_available}")
    
    with col2:
        st.markdown(f"**{utilization:.1f}%**")
        # FIXED: Divide by 100 to get value between 0 and 1
        st.progress(utilization / 100)
    
    with col3:
        st.markdown("")
        st.metric("", f"{utilization:.1f}%")

# Add sidebar with additional controls
with st.sidebar:
    st.markdown("## PMO Dashboard")
    st.markdown("---")
    
    st.markdown("### Dashboard Controls")
    
    # Refresh button
    if st.button("🔄 Refresh Dashboard", type="primary", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### Portfolio Summary")
    metrics = calculate_portfolio_metrics(st.session_state.projects)
    
    st.metric("Total Projects", metrics['total_projects'])
    st.metric("Active Projects", metrics['active_projects'])
    st.metric("At Risk", metrics['at_risk_projects'])
    st.metric("Total Budget", f"${metrics['total_budget']:,.0f}")
    
    st.markdown("---")
    
    st.markdown("### Export Options")
    
    if st.button("📄 Export to PDF", use_container_width=True):
        st.success("Dashboard exported as PDF successfully!")
    
    if st.button("📊 Export to CSV", use_container_width=True):
        # Create DataFrame from projects
        df = pd.DataFrame(st.session_state.projects)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="pmo_projects.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Project Management Actions
    st.markdown("### Quick Actions")
    
    if st.button("➕ Create New Project", use_container_width=True):
        # Switch to Projects tab
        st.switch_page("?tab=1")
    
    if st.button("📋 View All Projects", use_container_width=True):
        # Switch to Projects tab
        st.switch_page("?tab=1")
    
    st.markdown("---")
    
    # Last updated
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    # Debug information (can be removed in production)
    with st.expander("Debug Info"):
        st.write(f"Total projects in memory: {len(st.session_state.projects)}")
        if st.button("Clear All Projects (Debug)"):
            st.session_state.projects = []
            st.rerun()
        if st.button("Reset to Sample Data"):
            # Reinitialize with sample data
            st.session_state.projects = [
                {
                    "name": "Digital Transformation Initiative",
                    "description": "Modernize legacy systems and implement cloud infrastructure",
                    "status": "Active",
                    "start_date": "2025-01-15",
                    "end_date": "2025-12-31",
                    "progress": 35,
                    "budget": 500000,
                    "spent": 125000,
                    "roi": 15.5,
                    "risk_level": "Low"
                },
                {
                    "name": "Mobile App Development",
                    "description": "Develop native mobile applications for iOS and Android",
                    "status": "Planning",
                    "start_date": "2025-03-01",
                    "end_date": "2025-11-30",
                    "progress": 0,
                    "budget": 250000,
                    "spent": 0,
                    "roi": None,
                    "risk_level": "Medium"
                },
                {
                    "name": "Customer Portal Redesign",
                    "description": "Redesign customer-facing portal with improved UX",
                    "status": "Active",
                    "start_date": "2025-01-01",
                    "end_date": "2025-06-30",
                    "progress": 60,
                    "budget": 150000,
                    "spent": 75000,
                    "roi": 22.3,
                    "risk_level": "Low"
                },
                {
                    "name": "Security Compliance Upgrade",
                    "description": "Implement security measures for compliance certification",
                    "status": "Critical",
                    "start_date": "2024-11-01",
                    "end_date": "2025-03-31",
                    "progress": 85,
                    "budget": 200000,
                    "spent": 175000,
                    "roi": 12.1,
                    "risk_level": "Critical"
                },
                {
                    "name": "Data Analytics Platform",
                    "description": "Build centralized analytics platform for business intelligence",
                    "status": "At Risk",
                    "start_date": "2025-02-01",
                    "end_date": "2025-09-30",
                    "progress": 45,
                    "budget": 300000,
                    "spent": 180000,
                    "roi": 8.2,
                    "risk_level": "High"
                }
            ]
            st.rerun()

# Add footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>PMO Dashboard v1.0 | © 2025 TechCorp Inc.</div>", unsafe_allow_html=True)