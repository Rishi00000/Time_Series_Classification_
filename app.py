import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

from features import extract_features
from predict import predict, load_model

# Page configuration
st.set_page_config(
    page_title="Time Series Fault Classification",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #5a67d8;
        margin: 1rem 0;
        color: white;
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: white;
    }
    .warning-box {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
CLASS_MAP = {0: 'Normal', 1: 'Early Fault', 2: 'Fault'}
CLASS_NAMES = ['Normal', 'Early Fault', 'Fault']
CLASS_COLORS = {'Normal': '#2ecc71', 'Early Fault': '#f39c12', 'Fault': '#e74c3c'}

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

def plot_signal(signal, idx, true_label=None, pred_label=None, confidence=None):
    """Create an interactive plot for a single signal"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(signal))),
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color='#3498db', width=1.5)
    ))
    
    title = f"Signal #{idx}"
    if pred_label is not None:
        color = CLASS_COLORS.get(CLASS_MAP[pred_label], '#3498db')
        title += f" - Predicted: <b>{CLASS_MAP[pred_label]}</b>"
        if confidence is not None:
            title += f" (Confidence: {confidence:.2%})"
    if true_label is not None:
        title += f" | True: <b>{CLASS_MAP[true_label]}</b>"
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Steps",
        yaxis_title="Amplitude",
        height=300,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        template="plotly_white"
    )
    
    return fig

def plot_class_distribution(labels, title="Class Distribution"):
    """Plot class distribution"""
    class_counts = pd.Series([CLASS_MAP[l] for l in labels]).value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_counts.index,
            y=class_counts.values,
            marker_color=[CLASS_COLORS[name] for name in class_counts.index],
            text=class_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def plot_confidence_distribution(confidence, predictions):
    """Plot confidence score distribution by class"""
    df = pd.DataFrame({
        'Confidence': confidence,
        'Predicted Class': [CLASS_MAP[p] for p in predictions]
    })
    
    fig = px.box(
        df, 
        x='Predicted Class', 
        y='Confidence',
        color='Predicted Class',
        color_discrete_map=CLASS_COLORS,
        title="Confidence Score Distribution by Predicted Class"
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def display_metrics(y_true, y_pred):
    """Display classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h1>{accuracy:.2%}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Precision</h3>
            <h1>{precision:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Recall</h3>
            <h1>{recall:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1-Score</h3>
            <h1>{f1:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 Time Series Fault Classification System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>🎯 System Overview:</b> Advanced machine learning system for real-time fault detection in time series data.
        This system uses a stacked ensemble (Gradient Boosting + Logistic Regression) to classify signals into three categories:
        <b>Normal</b>, <b>Early Fault</b>, and <b>Fault</b>.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/maintenance.png", width=100)
        st.markdown("## 📊 Navigation")
        page = st.radio(
            "Select a page:",
            ["🏠 Home", "📈 Model Performance", "🔍 Test Your Data", "📚 About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎛️ Model Info")
        st.info("""
        **Architecture:**
        - Layer 1: Gradient Boosting
        - Layer 2: Logistic Regression
        - Features: 22 (Time + Frequency)
        - Signal Length: 1024 points
        """)
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "🔍 Test Your Data":
        show_test_page()
    elif page == "📚 About":
        show_about_page()

def show_home_page():
    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Check if model exists
    model_path = Path('model.pkl')
    if not model_path.exists():
        st.error("⚠️ Model file not found! Please train the model first by running `python train.py`")
        return
    
    # Load training data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 Training Data")
        try:
            train_data = np.load('data/train.npz')
            X_train, y_train = train_data['X'], train_data['y']
            
            st.metric("Total Samples", f"{len(y_train):,}")
            st.metric("Signal Length", f"{X_train.shape[1]:,} points")
            
            # Class distribution
            fig = plot_class_distribution(y_train, "Training Data Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Training data not found")
    
    with col2:
        st.markdown("### 📁 Validation Data")
        try:
            val_data = np.load('data/val.npz')
            X_val, y_val = val_data['X'], val_data['y']
            
            st.metric("Total Samples", f"{len(y_val):,}")
            st.metric("Signal Length", f"{X_val.shape[1]:,} points")
            
            # Class distribution
            fig = plot_class_distribution(y_val, "Validation Data Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Validation data not found")
    
    # Sample signals visualization
    st.markdown('<div class="sub-header">🔬 Sample Signals</div>', unsafe_allow_html=True)
    
    try:
        # Show one signal from each class
        for class_idx, class_name in CLASS_MAP.items():
            idx = np.where(y_val == class_idx)[0]
            if len(idx) > 0:
                sample_idx = idx[0]
                signal = X_val[sample_idx]
                fig = plot_signal(signal, sample_idx, true_label=class_idx)
                st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Unable to load sample signals")

def show_performance_page():
    st.markdown('<div class="sub-header">📈 Model Performance Evaluation</div>', unsafe_allow_html=True)
    
    # Data selection
    data_option = st.selectbox(
        "Select dataset to evaluate:",
        ["Validation Data", "Synthetic Test Data (100 samples)"]
    )
    
    data_path = 'data/val.npz' if data_option == "Validation Data" else 'data/synthetic_100.npz'
    
    try:
        # Load data
        data = np.load(data_path)
        X_raw = data['X']
        y_true = data['y']
        
        with st.spinner('Making predictions...'):
            # Make predictions
            pred_class, confidence, proba = predict(X_raw, 'model.pkl')
        
        st.markdown('<div class="success-box">✅ Predictions completed successfully!</div>', unsafe_allow_html=True)
        
        # Display metrics
        st.markdown("### 📊 Performance Metrics")
        display_metrics(y_true, pred_class)
        
        # Confusion Matrix and Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confusion Matrix")
            fig_cm = plot_confusion_matrix(y_true, pred_class)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Confidence Distribution")
            fig_conf = plot_confidence_distribution(confidence, pred_class)
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Detailed Classification Report
        st.markdown("### 📋 Detailed Classification Report")
        report = classification_report(y_true, pred_class, target_names=CLASS_NAMES, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(
            report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                          .format("{:.3f}", subset=['precision', 'recall', 'f1-score'])
                          .format("{:.0f}", subset=['support']),
            use_container_width=True
        )
        
        # Error Analysis
        st.markdown("### 🔍 Error Analysis")
        errors = y_true != pred_class
        error_count = errors.sum()
        
        if error_count > 0:
            st.warning(f"Found {error_count} misclassified samples ({error_count/len(y_true):.2%} error rate)")
            
            error_df = pd.DataFrame({
                'Sample ID': np.where(errors)[0],
                'True Label': [CLASS_MAP[y] for y in y_true[errors]],
                'Predicted Label': [CLASS_MAP[p] for p in pred_class[errors]],
                'Confidence': [f"{c:.2%}" for c in confidence[errors]]
            })
            
            st.dataframe(error_df, use_container_width=True)
            
            # Show misclassified signals
            if st.checkbox("Show misclassified signals"):
                error_indices = np.where(errors)[0]
                for idx in error_indices[:5]:  # Show first 5
                    fig = plot_signal(
                        X_raw[idx], 
                        idx, 
                        true_label=y_true[idx],
                        pred_label=pred_class[idx],
                        confidence=confidence[idx]
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 Perfect classification! No errors found.")
        
        # Download predictions
        st.markdown("### 💾 Download Predictions")
        results_df = pd.DataFrame({
            'sample_id': np.arange(len(pred_class)),
            'predicted_label': [CLASS_MAP[p] for p in pred_class],
            'true_label': [CLASS_MAP[y] for y in y_true],
            'confidence_score': np.round(confidence, 4),
            'p_normal': np.round(proba[:, 0], 4),
            'p_early_fault': np.round(proba[:, 1], 4),
            'p_fault': np.round(proba[:, 2], 4),
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
        
    except FileNotFoundError:
        st.error(f"⚠️ Data file not found: {data_path}")
    except Exception as e:
        st.error(f"⚠️ Error during evaluation: {str(e)}")

def show_test_page():
    st.markdown('<div class="sub-header">🔍 Test Your Own Data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>📤 Upload Instructions:</b>
        <ul>
            <li>Upload a <b>.npz</b> file containing your test data</li>
            <li>The file should contain an array 'X' with shape (N, 1024)</li>
            <li>Optionally, include 'y' array for true labels to evaluate accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an NPZ file",
        type=['npz'],
        help="Upload a .npz file containing time series data"
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            data = np.load(uploaded_file)
            
            if 'X' not in data:
                st.error("⚠️ The uploaded file must contain an 'X' array!")
                return
            
            X_raw = data['X']
            y_true = data['y'] if 'y' in data else None
            
            st.markdown('<div class="success-box">✅ File loaded successfully!</div>', unsafe_allow_html=True)
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{len(X_raw):,}")
            with col2:
                st.metric("Signal Length", f"{X_raw.shape[1]:,}")
            with col3:
                st.metric("Has True Labels", "Yes ✓" if y_true is not None else "No ✗")
            
            # Make predictions button
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner('Analyzing signals...'):
                    pred_class, confidence, proba = predict(X_raw, 'model.pkl')
                    st.session_state.uploaded_data = {
                        'X_raw': X_raw,
                        'y_true': y_true,
                        'pred_class': pred_class,
                        'confidence': confidence,
                        'proba': proba
                    }
                    st.session_state.predictions_made = True
                    st.rerun()
            
            # Display results if predictions were made
            if st.session_state.predictions_made and st.session_state.uploaded_data is not None:
                data = st.session_state.uploaded_data
                X_raw = data['X_raw']
                y_true = data['y_true']
                pred_class = data['pred_class']
                confidence = data['confidence']
                proba = data['proba']
                
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                # If true labels are available, show metrics
                if y_true is not None:
                    st.markdown("### 🎯 Performance Metrics")
                    display_metrics(y_true, pred_class)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_cm = plot_confusion_matrix(y_true, pred_class)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with col2:
                        fig_conf = plot_confidence_distribution(confidence, pred_class)
                        st.plotly_chart(fig_conf, use_container_width=True)
                
                # Prediction distribution
                st.markdown("### 📊 Prediction Distribution")
                fig_dist = plot_class_distribution(pred_class, "Predicted Class Distribution")
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Results table
                st.markdown("### 📋 Detailed Predictions")
                results_df = pd.DataFrame({
                    'Sample ID': np.arange(len(pred_class)),
                    'Predicted Label': [CLASS_MAP[p] for p in pred_class],
                    'Confidence': [f"{c:.2%}" for c in confidence],
                    'P(Normal)': [f"{p:.2%}" for p in proba[:, 0]],
                    'P(Early Fault)': [f"{p:.2%}" for p in proba[:, 1]],
                    'P(Fault)': [f"{p:.2%}" for p in proba[:, 2]],
                })
                
                if y_true is not None:
                    results_df.insert(1, 'True Label', [CLASS_MAP[y] for y in y_true])
                    results_df['Correct'] = ['✓' if p == t else '✗' for p, t in zip(pred_class, y_true)]
                
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Signal visualization
                st.markdown("### 🔬 Signal Visualization")
                num_signals = len(X_raw)
                
                if num_signals <= 20:
                    show_all = st.checkbox("Show all signals", value=False)
                    if show_all:
                        for idx in range(num_signals):
                            fig = plot_signal(
                                X_raw[idx],
                                idx,
                                true_label=y_true[idx] if y_true is not None else None,
                                pred_label=pred_class[idx],
                                confidence=confidence[idx]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        signal_idx = st.slider("Select signal to view:", 0, num_signals-1, 0)
                        fig = plot_signal(
                            X_raw[signal_idx],
                            signal_idx,
                            true_label=y_true[signal_idx] if y_true is not None else None,
                            pred_label=pred_class[signal_idx],
                            confidence=confidence[signal_idx]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    signal_idx = st.slider("Select signal to view:", 0, num_signals-1, 0)
                    fig = plot_signal(
                        X_raw[signal_idx],
                        signal_idx,
                        true_label=y_true[signal_idx] if y_true is not None else None,
                        pred_label=pred_class[signal_idx],
                        confidence=confidence[signal_idx]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.markdown("### 💾 Download Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name="test_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"⚠️ Error loading file: {str(e)}")
    
    # Reset button
    if st.session_state.predictions_made:
        if st.button("🔄 Upload New Data"):
            st.session_state.uploaded_data = None
            st.session_state.predictions_made = False
            st.rerun()

def show_about_page():
    st.markdown('<div class="sub-header">📚 About This System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This is an advanced **Time Series Fault Classification System** designed to detect and classify faults
    in time series signals using machine learning. The system is particularly useful for:
    
    - Predictive maintenance in industrial equipment
    - Early fault detection in machinery
    - Quality control in manufacturing processes
    - Anomaly detection in sensor data
    
    ## 🏗️ System Architecture
    
    ### Model Pipeline
    
    The system uses a **stacked ensemble approach**:
    
    1. **Feature Extraction** (22 features)
       - **Time Domain**: Mean, STD, RMS, Peak, Skewness, Kurtosis, Crest Factor, etc.
       - **Frequency Domain**: Spectral Centroid, Spectral Spread, Energy, Entropy, Band Powers
    
    2. **Layer 1: Gradient Boosting Classifier**
       - 100 estimators
       - Max depth: 4
       - Learning rate: 0.05
       - Generates leaf embeddings
    
    3. **Layer 2: Logistic Regression**
       - One-hot encoded leaf indices as input
       - Class weighting for imbalanced data
       - Softmax output for probability estimation
    
    ### Data Augmentation
    
    Training data is augmented using:
    - **Additive Gaussian noise** (2% of signal STD)
    - **Amplitude scaling** (0.9x to 1.1x)
    - **Time shifting** (±50 samples)
    
    ## 🎨 Class Definitions
    
    The system classifies signals into three categories:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: {CLASS_COLORS['Normal']}20; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {CLASS_COLORS['Normal']};">
            <h3 style="color: {CLASS_COLORS['Normal']};">✅ Normal</h3>
            <p>Healthy signal with no anomalies. Equipment operating within expected parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: {CLASS_COLORS['Early Fault']}20; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {CLASS_COLORS['Early Fault']};">
            <h3 style="color: {CLASS_COLORS['Early Fault']};">⚠️ Early Fault</h3>
            <p>Incipient fault detected. Requires monitoring and potential scheduled maintenance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: {CLASS_COLORS['Fault']}20; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {CLASS_COLORS['Fault']};">
            <h3 style="color: {CLASS_COLORS['Fault']};">🔴 Fault</h3>
            <p>Critical fault detected. Immediate attention required to prevent failure.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🛠️ Technical Stack
    
    - **Frontend**: Streamlit
    - **ML Framework**: Scikit-learn
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: NumPy, Pandas, SciPy
    
    ## 📊 Performance
    
    The model achieves high accuracy through:
    - Rich feature engineering (time + frequency domain)
    - Stacked ensemble architecture
    - Data augmentation for robustness
    - Class weighting for imbalanced data
    
    ## 🚀 Usage Guide
    
    1. **Model Performance**: View pre-evaluated results on validation/test data
    2. **Test Your Data**: Upload custom .npz files for real-time predictions
    3. **Visualization**: Interactive signal plots with predictions and confidence scores
    4. **Metrics**: Comprehensive evaluation with confusion matrices and classification reports
    
    ## 👨‍💻 Developer Information
    
    - **Repository**: [GitHub - 5_time_series_fault_classification](https://github.com/ReddyVikranth/5_time_series_fault_classification)
    - **Author**: ReddyVikranth
    
    ---
    
    <div class="info-box">
        <b>💡 Note:</b> This system is designed for demonstration and research purposes. 
        For production deployment, additional validation and safety measures should be implemented.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
