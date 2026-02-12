"""
Streamlit Frontend for Phishing Detector
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .phishing-alert {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #ccffcc;
        border-left: 5px solid #00ff00;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:5000"

if 'history' not in st.session_state:
    st.session_state.history = []


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def predict_url(url):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": url},
            timeout=10
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': 'Cannot connect to API. Please ensure Flask backend is running.'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def create_gauge_chart(probability):
    if probability >= 0.7:
        color = "red"
    elif probability >= 0.4:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Phishing Probability (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 70], 'color': '#FFD700'},
                {'range': [70, 100], 'color': '#FFB6C6'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_bar_chart(phishing_prob, legitimate_prob):
    fig = go.Figure(data=[
        go.Bar(
            x=['Legitimate', 'Phishing'],
            y=[legitimate_prob * 100, phishing_prob * 100],
            marker_color=['green', 'red'],
            text=[f'{legitimate_prob*100:.1f}%', f'{phishing_prob*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Classification",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        showlegend=False
    )
    return fig


def display_result(result):
    if not result['success']:
        st.error(f"❌ Error: {result['error']}")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if result['is_phishing']:
            st.markdown(f"""
            <div class="">
                <h2>⚠️ PHISHING DETECTED!</h2>
                <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                <p><strong>Confidence:</strong> {result['confidence']*100:.2f}%</p>
                <p>This URL appears to be phishing. Do not enter sensitive information.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="">
                <h2>✅ URL Appears Safe</h2>
                <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                <p><strong>Confidence:</strong> {result['confidence']*100:.2f}%</p>
                <p>This URL appears legitimate. However, always exercise caution.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Phishing Probability", f"{result['phishing_probability']*100:.2f}%")
        st.metric("Legitimate Probability", f"{result['legitimate_probability']*100:.2f}%")
    
    st.markdown("---")
    st.subheader("📊 Visual Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.plotly_chart(create_gauge_chart(result['phishing_probability']), use_container_width=True)
    
    with chart_col2:
        st.plotly_chart(create_bar_chart(result['phishing_probability'], result['legitimate_probability']), use_container_width=True)


def main():
    st.markdown('<p class="main-header">🔒 Phishing URL Detector</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Deep Learning with LSTM + Attention Mechanism</p>', unsafe_allow_html=True)
    
    api_status = check_api_health()
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("LSTM + Attention mechanism for phishing detection")
        
        st.markdown("---")
        st.header("🔌 API Status")
        if api_status:
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
            st.warning("Start backend:\n```python backend/app.py```")
        
        st.markdown("---")
        st.header("📊 Statistics")
        if st.session_state.history:
            total = len(st.session_state.history)
            phishing = sum(1 for h in st.session_state.history if h.get('is_phishing', False))
            st.metric("Total Scans", total)
            st.metric("Phishing Detected", phishing)
            st.metric("Safe URLs", total - phishing)
        else:
            st.info("No scans yet")
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    
    if not api_status:
        st.error("⚠️ Cannot connect to API. Please start the Flask backend: `python backend/app.py`")
        return
    
    st.header("🔍 Check a URL")
    
    url_input = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and url_input:
        with st.spinner("Analyzing URL..."):
            result = predict_url(url_input)
            
            if result['success']:
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.insert(0, result)
                display_result(result)
            else:
                st.error(f"❌ Error: {result['error']}")
    elif analyze_button:
        st.warning("⚠️ Please enter a URL to analyze")
    
    # Example URLs
    st.markdown("---")
    st.subheader("📝 Try Example URLs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Legitimate:**")
        if st.button("https://www.google.com"):
            st.session_state.example_url = "https://www.google.com"
            st.rerun()
        if st.button("https://www.github.com"):
            st.session_state.example_url = "https://www.github.com"
            st.rerun()
    
    with col2:
        st.markdown("**Suspicious:**")
        if st.button("http://paypal-secure.tk/login"):
            st.session_state.example_url = "http://paypal-secure.tk/login"
            st.rerun()
    
    # History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Scan History")
        
        history_df = pd.DataFrame([
            {
                'Time': h['timestamp'],
                'URL': h['url'][:50] + '...' if len(h['url']) > 50 else h['url'],
                'Result': h['prediction'],
                'Probability': f"{h['phishing_probability']*100:.1f}%",
                'Risk': h['risk_level']
            }
            for h in st.session_state.history[:10]
        ])
        
        st.dataframe(history_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
