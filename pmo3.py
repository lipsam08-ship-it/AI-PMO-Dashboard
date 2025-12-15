"""
Simple AI-Powered PMO Dashboard
File: simple_pmo_dashboard.py
Run: streamlit run simple_pmo_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page setup
st.set_page_config(page_title="Simple PMO Dashboard", layout="wide")

# Title
st.title("📊 Simple PMO Dashboard")
st.markdown("---")

# ====================
# SIDEBAR - DATA INPUT
# ====================
with st.sidebar:
    st.header("📥 Data Input")
    
    # Option 1: Upload file
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    # Option 2: Sample data
    use_sample = st.checkbox("Use sample data")
    
    # Option 3: Manual input
    if st.button("Enter data manually"):
        st.session_state.manual_input = True
    
    st.markdown("---")
    st.header("🤖 AI Settings")
    
    # OpenAI API Key
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("✅ API key saved")
    
    st.markdown("---")
    
    if st.button("Generate AI Report"):
        st.session_state.generate_report = True

# ====================
# LOAD DATA
# ====================
def load_data():
    """Load data from uploaded file or create sample"""
    
    if uploaded_file is not None:
        # Load uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    
    elif use_sample:
        # Create sample data
        sample_data = {
            'Project': ['Website Redesign', 'Mobile App', 'Cloud Migration', 'Data Analytics'],
            'Health Score': [85, 45, 70, 90],
            'Budget Used %': [65, 110, 80, 60],
            'Timeline %': [80, 40, 70, 95],
            'Risk': ['Low', 'High', 'Medium', 'Low'],
            'Team Size': [5, 8, 12, 6],
            'Department': ['Marketing', 'Engineering', 'IT', 'Analytics']
        }
        return pd.DataFrame(sample_data)
    
    else:
        # Return empty if no data
        return pd.DataFrame()

# Load data
df = load_data()

# ====================
# MANUAL INPUT
# ====================
if 'manual_input' in st.session_state and st.session_state.manual_input:
    st.subheader("✍️ Add Project")
    
    with st.form("add_project"):
        col1, col2 = st.columns(2)
        
        with col1:
            project = st.text_input("Project Name")
            department = st.selectbox("Department", ["IT", "Engineering", "Marketing", "Finance"])
            health = st.slider("Health Score", 0, 100, 75)
        
        with col2:
            budget = st.slider("Budget Used %", 0, 150, 50)
            timeline = st.slider("Timeline %", 0, 100, 50)
            team = st.number_input("Team Size", 1, 50, 5)
        
        submit = st.form_submit_button("Add Project")
        
        if submit and project:
            # Add to dataframe
            new_row = {
                'Project': project,
                'Health Score': health,
                'Budget Used %': budget,
                'Timeline %': timeline,
                'Risk': 'High' if health < 60 or budget > 90 else ('Medium' if health < 75 or budget > 75 else 'Low'),
                'Team Size': team,
                'Department': department
            }
            
            if df.empty:
                df = pd.DataFrame([new_row])
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            st.success(f"Added {project}")
            st.session_state.manual_input = False
            st.rerun()

# ====================
# DASHBOARD MAIN CONTENT
# ====================

if df.empty:
    st.info("Please upload data or use sample data from the sidebar.")
    st.stop()

# Display data summary
st.subheader("📋 Project Data")
st.dataframe(df, use_container_width=True)

# ====================
# KPIs
# ====================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Projects", len(df))

with col2:
    high_risk = len(df[df['Risk'] == 'High'])
    st.metric("High Risk", high_risk)

with col3:
    avg_health = df['Health Score'].mean()
    st.metric("Avg Health", f"{avg_health:.1f}%")

with col4:
    avg_budget = df['Budget Used %'].mean()
    st.metric("Avg Budget", f"{avg_budget:.1f}%")

# ====================
# VISUALIZATIONS
# ====================
st.subheader("📈 Charts")

# Chart 1: Health Scores
fig1 = px.bar(
    df, 
    x='Project', 
    y='Health Score',
    color='Risk',
    title='Project Health Scores',
    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
)
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Budget vs Health
col1, col2 = st.columns(2)

with col1:
    fig2 = px.scatter(
        df,
        x='Budget Used %',
        y='Health Score',
        color='Risk',
        size='Team Size',
        hover_data=['Project'],
        title='Budget vs Health',
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Risk distribution
    risk_counts = df['Risk'].value_counts()
    fig3 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Risk Distribution',
        color=risk_counts.index,
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ====================
# AI INSIGHTS
# ====================
st.subheader("🤖 AI Insights")

if 'openai_key' in st.session_state:
    if st.session_state.get('generate_report', False):
        with st.spinner("Generating AI insights..."):
            try:
                import openai
                
                # Prepare data summary
                summary = f"""
                Total Projects: {len(df)}
                High Risk Projects: {high_risk}
                Average Health Score: {avg_health:.1f}%
                Average Budget Used: {avg_budget:.1f}%
                
                Projects:
                {df.to_string()}
                """
                
                # Call OpenAI
                openai.api_key = st.session_state.openai_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a PMO analyst. Give concise insights."},
                        {"role": "user", "content": f"Analyze this project data and give 3 key insights:\n{summary}"}
                    ],
                    max_tokens=200
                )
                
                insights = response.choices[0].message.content
                
                st.success("✅ AI Insights Generated")
                st.info(insights)
                
                st.session_state.generate_report = False
                
            except Exception as e:
                st.error(f"AI Error: {str(e)}")
    
    else:
        if st.button("Generate AI Report"):
            st.session_state.generate_report = True
            st.rerun()

else:
    # Simple insights without AI
    st.info("Simple Analysis:")
    
    if high_risk > 0:
        st.error(f"🚨 {high_risk} projects are at HIGH risk")
    
    if avg_budget > 100:
        st.warning(f"⚠️ Average budget is {avg_budget:.1f}% (over budget)")
    
    if avg_health < 70:
        st.warning(f"⚠️ Average health score is {avg_health:.1f}% (needs improvement)")
    else:
        st.success(f"✅ Average health score is {avg_health:.1f}% (good)")
    
    # Recommendations
    st.info("📋 Recommendations:")
    if high_risk > 0:
        st.write("• Focus on high-risk projects first")
    if avg_budget > 100:
        st.write("• Review budget allocations")
    st.write("• Schedule weekly review meetings")

# ====================
# EXPORT OPTION
# ====================
st.markdown("---")
st.subheader("📤 Export")

if st.button("Download Data as CSV"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Click to download",
        data=csv,
        file_name=f"pmo_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Simple PMO Dashboard v1.0 | Upload data to get started")