
# 💧 Water Quality Prediction using Machine Learning

An interactive machine learning application that predicts water potability based on key chemical properties. Built using Python and Streamlit as part of my graduation project at Yerevan State University.

## 🚀 Live Demo
🔗 [Try the App Here](https://water-quality-prediction-25.streamlit.app/)
> No installation needed to try the app — just click the link above!
>

## 📌 Key Features
- **6 ML Models**: SVM, Random Forest, Logistic Regression, Decision Tree, KNN, Naive Bayes
- **Data Preprocessing**: Advanced missing value imputation
- **Real-Time Predictions**: Instant water safety classification
- **Visual Analytics**: Interactive EDA and model comparisons

## 🛠️ Tech Stack
```python
# Core Dependencies
pandas==1.3.4
scikit-learn==0.24.2
streamlit==0.88.0
matplotlib==3.4.3
seaborn==0.11.2
jupyter==1.0.0
joblib==1.0.1
```

## 📊 Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM | 67.78% | 76.30% | 25.20% | 37.88% |
| Random Forest | 68.21% | 60.88% | 30.20% | 40.38% |
| Logistic Regression | 61.05% | 66.67% | 0.31% | 0.62% |
| Decision Tree | 56.41% | 45.32% | 43.97% | 44.64% |
| K-Nearest Neighbors | 60.32% | 48.88% | 37.40% | 42.38% |
| Naive Bayes | 61.97% | 53.17% | 20.97% | 30.08% |

## 📂 Project Files
```
WaterQualityPrediction/
├── WaterQualityPrediction.ipynb  # Complete analysis notebook
├── prediction.py                # Prediction script
├── svm.pkl                      # Trained SVM model
├── scaler.pkl                   # Feature scaler
└── water_potability.csv         # Dataset (3276 samples)
```

## 🔧 How To Use
1. Clone repository:
```bash
git clone https://github.com/mkrtchyyan/WQ_YSU.git
cd WQ_YSU
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make predictions:
```python
import joblib

# Load model and scaler
model = joblib.load('svm.pkl')
scaler = joblib.load('scaler.pkl')

# Sample input (replace with your values)
sample = [[7.08, 196, 22014, 7.12, 334, 426, 14.3, 66.4, 3.97]]

# Predict
scaled = scaler.transform(sample)
result = "Safe" if model.predict(scaled)[0] == 1 else "Unsafe"
print(f"Water is: {result}")
```

## 🌊 Dataset Features
| Parameter | Unit | Description |
|-----------|------|-------------|
| pH | 0-14 | Acidity level |
| Hardness | mg/L | Calcium carbonate |
| Solids | ppm | Total dissolved solids |
| Chloramines | ppm | Disinfectant |
| Sulfate | mg/L | SO₄ concentration |
| Conductivity | μS/cm | Electrical conductivity |
| Organic_carbon | ppm | Carbon content |
| Trihalomethanes | μg/L | Chemical byproduct |
| Turbidity | NTU | Water clarity |

## 📜 License
MIT License © 2025 - Yerevan State University

## 📬 Contact
**Developer**: Manan Mkrtchyan <br>
**Email**: manan.mkrtchyan.02@gmail.com <br>
**GitHub**: [@mkrtchyyan](https://github.com/mkrtchyyan)
```
