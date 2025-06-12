# Resume Screening App v2 

**Resume Classification using Machine Learning & Real-World Data**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.1+-orange.svg)](https://scikit-learn.org)

A machine learning application project that automatically classifies resumes into job categories using real-world data from Kaggle. Built with Python, Streamlit and powered by NLP techniques.

## Key Features

### Core Functionality
- **ML Classification**: Uses trained ML models (SVM/Random Forest) with TF-IDF vectorization
- **Real Data Training**: Trained on actual resume dataset from Kaggle 
- **Multi-Format Support**: Handles PDF, DOCX and TXT files seamlessly. Single and Multiple file uploading and analyzing
- **Real-Time Processing**: Instant resume analysis with confidence scoring
- **Interactive Visualizations**: Probability breakdowns and detailed analytics using Plotly

### Advanced Features
- **Batch Processing**: Analyze multiple resumes simultaneously
- **Comprehensive Reports**: Detailed classification reports with downloadable CSV results
- **Confidence Scoring**: Know how certain the model is about each prediction
- **Smart Text Preprocessing**: Advanced text cleaning and NLP preprocessing
- **Responsive UI**: Modern, mobile-friendly interface built with Streamlit

## Live Demo

Experience the app in action: [Coming Soon -> Deployment in Progress]

## Current Performance Metrics

| Metric | Value | Status |
|--------|--------|--------|
| **Test Accuracy** | **66.94%** | 🔄 *Under Enhancement* |
| **Training Data** | Kaggle Resume Dataset | ✅ Real-world data |
| **Categories Supported** | 25+ Job Categories | ✅ Comprehensive |
| **File Formats** | PDF, DOCX, TXT | ✅ Multi-format |

> **Note**: Actively working on improving accuracy through advanced NLP techniques, ensemble methods and hyperparameter optimization. Target accuracy: 95%+

## 🏗️ Architecture Overview

```
├── Frontend (Streamlit)
│   ├── Single Resume Analysis
│   ├── Batch Processing
│   └── Interactive Visualizations
│
├── ML Pipeline
│   ├── Data Preprocessing
│   ├── TF-IDF Vectorization
│   ├── SVM/Random Forest Classification
│   └── Confidence Scoring
│
├── Data Processing
│   ├── Multi-format File Handling
│   ├── Text Cleaning & NLP
│   └── Feature Engineering
│
└── Backend Services
    ├── Model Training & Validation
    ├── Performance Metrics
    └── Batch Processing Engine
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager 
- 4GB+ RAM recommended
- Requirements.txt

### Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nitya-01/Resume-Screening-v2.git
   cd Resume-Screening-v2
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv resume_env
   
   # On Windows
   resume_env\Scripts\activate
   
   # On macOS/Linux
   source resume_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Data Directory**
   ```bash
   mkdir data models
   ```

5. **Download Dataset** (Optional - for training)
   ```bash
   # Place your resume dataset CSV in the data/ directory
   # Expected format: Resume text column + Category column
   ```

## Usage Guide

### Option 1: Use Pre-trained Model (Recommended)
If you have a pre-trained model, place the model files in the `models/` directory and skip to step 3.

### Option 2: Train Your Own Model

1. **Prepare Your Data**
   - Place your resume dataset (in CSV format) in the `data/` directory
   - Ensure it has columns for resume text and job categories

2. **Train the Model**
   ```bash
   python train_real_model.py
   ```
   This will:
   - Process and clean the resume data
   - Train the ML model
   - Save the trained model components
   - Display performance metrics

3. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

4. **Access the App**
   - Open your browser to `http://localhost:8501`
   - Start analyzing resumes!

## Project Structure

```
resume-screening-app-v2/
│
├──  app.py                    # Main Streamlit application
├──  train_real_model.py       # Model training script
├──  data_loader.py            # Data processing utilities
├──  requirements.txt          # Python dependencies
├──  README.md                 # This file
│
├──   data/                     # Dataset directory
│   ├── Resume.csv              # Raw resume dataset
│   ├── label_encoder.pkl       # Encoding techniques
│   └── processed_resumes.csv   # Processed data
│
└──   models/                   # Trained model storage
    ├── resume_classifier.pkl   # Trained classifier
    ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
    ├── category_mapping.pkl    # Label mappings
    └── label_encoder.pkl       # Performance metrics
```

## Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Removal of PII (emails, phone numbers)
   - Stop word removal and tokenization
   - Minimum sample filtering per category

2. **Feature Engineering**
   - TF-IDF vectorization with n-grams (1,2)
   - Maximum 5,000 features
   - Minimum document frequency filtering
   - Maximum document frequency capping (95%)

3. **Model Training**
   - **Primary**: Support Vector Machine (SVM) with linear kernel
   - **Alternative**: Random Forest Classifier
   - Cross-validation for model selection
   - Hyperparameter optimization

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix analysis
   - Per-category performance metrics
   - Confidence score calibration

### Supported Job Categories

The model currently supports 23 job categories including:
- Engineering & Development
- Data Science & Machine Learning
- Digital Marketing & Sales
- Healthcare & Medical
- Finance & Banking
- Education & Academia
- Human Resources
- Operations & Management
- Agriculture
- and many more...

## Performance Analysis

### Current Status (v2.0)
- **Accuracy**: 66.94% (actively improving)
- **Training Time**: ~2-5 minutes on standard hardware
- **Inference Speed**: <1 second per resume
- **Memory Usage**: ~500MB during training, ~100MB during inference

### Improvement Roadmap
- [ ] **Target Accuracy**: 95%+ through advanced techniques
- [ ] **Transformer Models**: BERT/RoBERTa integration
- [ ] **Ensemble Methods**: Multiple model combination
- [ ] **Active Learning**: Continuous improvement pipeline
- [ ] **Hyperparameter Tuning** 

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Error: Missing model files
# Solution: Train the model first
python train_real_model.py
```

**2. Data Loading Issues**
```bash
# Error: CSV format not recognized
# Solution: Ensure your CSV has resume text and category columns
# Check column names and data format
```

**3. Memory Issues**
```bash
# Error: Out of memory during training
# Solution: Reduce max_features in TfidfVectorizer
# Or use a smaller dataset for training
```

## Future Enhancements

- [ ] **Accuracy Improvements**: Target 85%+ accuracy
- [ ] **Advanced NLP**: Transformer model integration
- [ ] **UI integration and enhancements**: Better mobile responsiveness
- [ ] **API Development**: REST API for integration
- [ ] **Multi-language Support**: Support for non-English resumes
- [ ] **Skill Extraction**: Automatic skill identification
- [ ] **Salary Prediction**: Integration with salary data
- [ ] **ATS Compliance**: Resume optimization for ATS systems
- [ ] **Cloud Deployment**: AWS/GCP deployment
- [ ] **Enterprise Features**: Multi-tenant support
- [ ] **Advanced Analytics**: Comprehensive dashboard
- [ ] **Mobile Apps**: Native Android/iOS applications

## 🤝 Contributing

I welcome contributions! Here's how you can help:

### Ways to Contribute
1. **Bug Reports**: Report issues via GitHub issues or 'Contact me' page 
2. **Feature Requests**: Suggest new features
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve docs and examples
5. **Testing**: Help with testing and validation

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests if applicable
5. Submit pull request

## 👩‍💻 Contact & Support

**Nitya Gupta**  
**Email**: [guptanitya.147@gmail.com](mailto:guptanitya.147@gmail.com) 
**LinkedIn**: [Connect on LinkedIn](https://linkedin.com/in/nitya-gupta-66361128a/)  
**GitHub**: [View GitHub Profile](https://github.com/Nitya-01)  

### Coding Standards
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Include unit tests for new features
- Update documentation as needed

## Educational Purpose
This project demonstrates:

- End-to-end ML pipeline development
- Real-world data processing techniques
- Web application development with Streamlit
- NLP and text classification methods
- Software engineering best practices

## Dataset Attribution

This project uses the Resume Dataset from Kaggle:
- **Source**: [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Usage**: Educational and research purposes

**Let's connect and build amazing solutions together!**

</div>
