# 🔧 Time Series Fault Classification - Streamlit UI

A professional web interface for the Time Series Fault Classification system, built with Streamlit.

## 🌟 Features

### 📊 Home Page
- Dataset overview with statistics
- Training and validation data distributions
- Sample signal visualizations from each class

### 📈 Model Performance
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1-Score)
- Interactive confusion matrix
- Confidence score distribution analysis
- Detailed classification reports
- Error analysis with misclassified samples
- Downloadable prediction results

### 🔍 Test Your Data
- Upload custom test data (.npz format)
- Real-time predictions with confidence scores
- Interactive signal visualization
- Performance evaluation (if true labels provided)
- Downloadable results in CSV format

### 📚 About
- System architecture documentation
- Feature engineering details
- Model pipeline explanation
- Technical stack information

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed and the model is trained.

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Train the Model (if not already trained)

```bash
python train.py
```

This will:
- Load training and validation data from `data/` folder
- Apply data augmentation
- Train the stacked ensemble model
- Save the model to `model.pkl`

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📁 Data Format

### For Testing Your Own Data

Upload `.npz` files with the following format:

```python
# Required
X: numpy array of shape (N, 1024)  # N samples, each with 1024 time steps

# Optional (for accuracy evaluation)
y: numpy array of shape (N,)  # True labels (0=Normal, 1=Early Fault, 2=Fault)
```

### Example Data Creation

```python
import numpy as np

# Create synthetic test data
X = np.random.randn(100, 1024)  # 100 samples
y = np.random.randint(0, 3, 100)  # 100 labels

# Save as NPZ
np.savez('my_test_data.npz', X=X, y=y)
```

## 🎨 UI Components

### Metrics Dashboard
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted harmonic mean

### Interactive Visualizations
- **Signal Plots**: Interactive time series with Plotly
- **Confusion Matrix**: Heatmap showing prediction vs true labels
- **Class Distribution**: Bar charts for data balance
- **Confidence Distribution**: Box plots by predicted class

### Color Coding
- 🟢 **Normal**: Green (#2ecc71)
- 🟠 **Early Fault**: Orange (#f39c12)
- 🔴 **Fault**: Red (#e74c3c)

## 🛠️ Model Architecture

1. **Feature Extraction** (22 features)
   - Time domain: Mean, STD, RMS, Peak, Skewness, Kurtosis, etc.
   - Frequency domain: Spectral features using FFT

2. **Layer 1: Gradient Boosting**
   - Generates leaf node embeddings from features

3. **Layer 2: Logistic Regression**
   - One-hot encoded leaves → class probabilities

## 📊 Available Datasets

- `data/train.npz` - Training dataset
- `data/val.npz` - Validation dataset
- `data/synthetic_100.npz` - Test dataset (100 samples)

## 💡 Tips

1. **Performance Page**: Select different datasets to compare model performance
2. **Test Page**: Upload your own .npz files to get real-time predictions
3. **Signal Visualization**: Use the slider to browse through individual signals
4. **Download Results**: Export predictions as CSV for further analysis

## 🔧 Troubleshooting

### Model Not Found
```bash
# Train the model first
python train.py
```

### Data Files Missing
Ensure the `data/` folder contains:
- `train.npz`
- `val.npz`
- `synthetic_100.npz` (optional)

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📈 Performance Benchmarks

Typical performance on validation data:
- **Accuracy**: 95-98%
- **Precision**: 94-97%
- **Recall**: 95-98%
- **F1-Score**: 95-97%

## 🤝 Contributing

This UI is designed to be extensible. Feel free to add:
- Additional visualization types
- More detailed analytics
- Export options (PDF reports, etc.)
- Real-time signal streaming

## 📝 License

Same as the main project.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Scikit-learn](https://scikit-learn.org/) - Machine learning

---

**Enjoy fault hunting! 🔍🔧**
