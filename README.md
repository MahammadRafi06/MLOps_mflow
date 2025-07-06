# Loan Prediction MLflow Platform

A comprehensive machine learning platform for loan approval prediction featuring advanced experiment tracking, model comparison, and automated hyperparameter optimization using MLflow on Databricks.

## Overview

This project implements an end-to-end machine learning solution for loan approval prediction, leveraging multiple algorithms and sophisticated experiment tracking. The platform provides robust model comparison capabilities, automated hyperparameter tuning, and comprehensive performance analysis to help financial institutions make data-driven lending decisions.

## Key Features

### Machine Learning Pipeline
- **Multi-Algorithm Approach**: Implements Random Forest, Logistic Regression, and Decision Tree classifiers
- **Automated Preprocessing**: Intelligent handling of missing values and categorical encoding
- **Feature Engineering**: Domain-specific feature creation and transformation
- **Hyperparameter Optimization**: Grid search cross-validation for optimal model performance

### Experiment Management
- **MLflow Integration**: Comprehensive experiment tracking and model registry
- **Automated Logging**: Parameters, metrics, and artifacts automatically captured
- **Model Comparison**: Side-by-side performance analysis across algorithms
- **Version Control**: Systematic model versioning and artifact management

### Data Processing
- **Smart Imputation**: Median imputation for numerical features, mode for categorical
- **One-Hot Encoding**: Categorical feature transformation for model compatibility
- **Outlier Treatment**: Quantile-based outlier clipping for improved model stability
- **Feature Scaling**: Log transformation for skewed numerical features

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │   Preprocessing │    │   Feature       │
│   - CSV File    │ -> │   - Imputation  │ -> │   Engineering   │
│   - Loan Data   │    │   - Encoding    │    │   - Transforms  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model         │    │   Hyperparameter│    │   Data Split    │
│   Comparison    │ <- │   Optimization  │ <- │   - Train/Test  │
│   - Metrics     │    │   - Grid Search │    │   - Validation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │
          v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   Model         │    │   Performance   │
│   - Tracking    │    │   Registry      │    │   Analysis      │
│   - Artifacts   │    │   - Versioning  │    │   - ROC Curves  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technology Stack

### Core ML Framework
- **Python 3.10+**: Primary programming language
- **Scikit-learn**: Machine learning algorithms and utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Experiment Tracking
- **MLflow**: Comprehensive ML lifecycle management
- **Databricks**: Unified analytics platform for ML workflows
- **Matplotlib**: Visualization and ROC curve generation
- **Jupyter Notebooks**: Interactive development environment

### Data Processing
- **Pandas**: Data loading, cleaning, and transformation
- **Scikit-learn Preprocessing**: Imputation and encoding utilities
- **NumPy**: Mathematical operations and transformations

## Dataset

### Loan Dataset Features
- **Loan_ID**: Unique identifier for each loan application
- **Gender**: Applicant's gender (Male/Female)
- **Married**: Marital status (Yes/No)
- **Dependents**: Number of dependents
- **Education**: Education level (Graduate/Not Graduate)
- **Self_Employed**: Employment type (Yes/No)
- **ApplicantIncome**: Primary applicant's income
- **CoapplicantIncome**: Co-applicant's income
- **LoanAmount**: Loan amount requested
- **Loan_Amount_Term**: Loan term in months
- **Credit_History**: Credit history availability
- **Property_Area**: Property location (Urban/Semiurban/Rural)
- **Loan_Status**: Target variable (Y/N)

### Data Preprocessing Pipeline
1. **Missing Value Handling**: Smart imputation based on feature type
2. **Categorical Encoding**: One-hot encoding for categorical features
3. **Outlier Treatment**: Quantile-based clipping (5th-95th percentile)
4. **Feature Engineering**: Log transformation and income aggregation
5. **Target Encoding**: Label encoding for binary classification

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- MLflow 2.16.2 or compatible version
- Access to Databricks workspace (optional)
- Jupyter Notebook environment

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/loan-prediction-mlflow.git
   cd loan-prediction-mlflow
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   # Create conda environment (alternative)
   conda env create -f conda.yaml
   conda activate Loan_prediction
   ```

### MLflow Configuration

1. **Local MLflow Server** (Optional)
   ```bash
   # Start MLflow tracking server
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. **Databricks Integration**
   ```python
   import mlflow
   mlflow.set_tracking_uri("databricks")
   mlflow.set_experiment("/Loan_exp")
   ```

## Usage

### Running the Complete Pipeline

1. **Execute main application**
   ```bash
   python app.py
   ```

2. **Run Jupyter experiments**
   ```bash
   jupyter notebook experiments.ipynb
   ```

### Model Training & Evaluation

The platform automatically trains three different models:

1. **Random Forest Classifier**
   ```python
   param_grid_forest = {
       'n_estimators': [200, 400, 700],
       'max_depth': [10, 20, 30],
       'criterion': ["gini", "entropy"],
       'max_leaf_nodes': [50, 100]
   }
   ```

2. **Logistic Regression**
   ```python
   param_grid_log = {
       'C': [100, 10, 1.0, 0.1, 0.01],
       'penalty': ['l1', 'l2'],
       'solver': ['liblinear']
   }
   ```

3. **Decision Tree Classifier**
   ```python
   param_grid_tree = {
       "max_depth": [3, 5, 7, 9, 11, 13],
       'criterion': ["gini", "entropy"]
   }
   ```

### MLflow Experiment Tracking

Each model run automatically logs:
- **Parameters**: All hyperparameters from grid search
- **Metrics**: Accuracy, F1-score, AUC, and CV scores
- **Artifacts**: ROC curves, model binaries, and evaluation plots
- **Model Registry**: Versioned models for deployment

### Accessing Results

1. **MLflow UI**
   ```bash
   mlflow ui --port 5000
   ```

2. **Databricks Experiments**
   - Navigate to Databricks workspace
   - Access "Experiments" section
   - Review `/Loan_exp` experiment runs

## Model Performance

### Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Cross-Validation Score**: 5-fold CV performance

### Model Comparison

The platform provides comprehensive comparison across all three algorithms:

```python
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    return accuracy, f1, auc
```

### Feature Importance Analysis

- **Random Forest**: Built-in feature importance scores
- **Logistic Regression**: Coefficient analysis
- **Decision Tree**: Feature importance based on impurity reduction

## Advanced Features

### Automated Hyperparameter Tuning
- **Grid Search CV**: Exhaustive search over parameter combinations
- **5-Fold Cross-Validation**: Robust performance estimation
- **Best Parameter Logging**: Automatic selection and logging of optimal parameters

### Data Quality Enhancements
- **Outlier Detection**: Statistical outlier identification and treatment
- **Missing Value Analysis**: Comprehensive missing data handling
- **Feature Correlation**: Multicollinearity detection and treatment

### Model Interpretability
- **ROC Curve Analysis**: Visual performance assessment
- **Feature Importance**: Understanding key predictive factors
- **Prediction Confidence**: Model certainty analysis

## Project Structure

```
loan-prediction-mlflow/
├── app.py                    # Main application script
├── experiments.ipynb         # Jupyter notebook for experimentation
├── train.csv                 # Training dataset
├── conda.yaml               # Conda environment specification
├── MLProject                 # MLflow project configuration
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── plots/                    # Generated visualizations
│   └── ROC_curve.png        # ROC curve plots
└── mlruns/                  # MLflow experiment tracking
    └── experiments/         # Experiment artifacts
```

## API Reference

### Core Functions

#### Data Loading
```python
def load_data():
    """Load and preprocess the loan dataset"""
    # Returns preprocessed X_train, X_test, y_train, y_test
```

#### Model Training
```python
def forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning"""
    
def lr(X_train, y_train):
    """Train Logistic Regression with hyperparameter tuning"""
    
def tree(X_train, y_train):
    """Train Decision Tree with hyperparameter tuning"""
```

#### Experiment Logging
```python
def mlflow_logging(model, X, y, name):
    """Log model performance and artifacts to MLflow"""
```

## Best Practices

### Code Organization
- Modular function design for reusability
- Clear separation of data processing and modeling
- Comprehensive error handling and logging

### Experiment Management
- Consistent naming conventions for experiments
- Comprehensive artifact logging
- Version control integration

### Performance Optimization
- Efficient data processing pipelines
- Optimized hyperparameter search spaces
- Memory-efficient model training

## Contributing

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/model-improvement`)
3. Implement changes with appropriate tests
4. Update documentation as needed
5. Submit a pull request with detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality
- Maintain MLflow experiment consistency

## Troubleshooting

### Common Issues

1. **MLflow Connection Issues**
   ```bash
   # Check MLflow server status
   mlflow server --help
   
   # Verify Databricks connection
   databricks configure --token
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Implement data chunking
   chunk_size = 1000
   for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
       # Process chunk
   ```

3. **Model Performance Issues**
   - Check data quality and preprocessing steps
   - Verify hyperparameter search spaces
   - Review feature engineering pipeline

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MLflow**: For comprehensive ML lifecycle management
- **Databricks**: For unified analytics platform
- **Scikit-learn**: For machine learning algorithms
- **Pandas**: For data manipulation capabilities

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Review MLflow documentation for tracking issues
- Contact the development team
- Check Databricks documentation for platform-specific help

---

*This project demonstrates advanced MLOps practices for financial services. Always ensure proper validation and regulatory compliance for production lending applications.*