# DSND Pipelines Project

This repository contains all the code, data, and results for the DSND Pipelines project. The project aims to predict customer recommendations based on textual reviews and associated metadata, leveraging advanced Natural Language Processing (NLP) techniques and machine learning models.

## Project Structure

```
DSND-PIPELINES-PROJECT/
├── .vscode/                        # VSCode settings
├── data/
│   └── reviews.csv                 # Raw data file
├── Class 0 Optimized_classification_report.csv
├── Class 0 Optimized_confusion_matrix.csv
├── Class 1 Optimized_classification_report.csv
├── Class 1 Optimized_confusion_matrix.csv
├── Gradient Boosting_classification_report.csv
├── Gradient Boosting_confusion_matrix.csv
├── gradient_boosting_model.pkl     # Gradient Boosting model pickle file
├── Layered Inference_classification_report.csv
├── Layered Inference_confusion_matrix.csv
├── Logistic Regression_classification_report.csv
├── Logistic Regression_confusion_matrix.csv
├── logistic_regression_model.pkl   # Logistic Regression model pickle file
├── README.md                       # Project README file
├── starter.ipynb                   # Jupyter notebook with code and analysis
├── .gitignore                      # Files to ignore in version control
├── CODEOWNERS                      # Code ownership details
├── LICENSE.txt                     # License details
└── requirements.txt                # Python dependencies
```

## Introduction

The DSND Pipelines Project predicts whether customers will recommend products based on their reviews and metadata. By applying machine learning models, such as Gradient Boosting and Logistic Regression, the project delivers actionable insights for customer satisfaction and product improvement.

## Getting Started

Follow these instructions to set up and run the project locally.

---

### **2. Project Instructions**

### Project Instructions

This project evaluates customer reviews to predict product recommendations using a machine learning pipeline. Follow these instructions to replicate the project:

1. **Set Up the Environment**:
   - Clone the repository and set up a virtual environment.
   - Install the required dependencies using `pip install -r requirements.txt`.

2. **Run the Notebook**:
   - Open `starter.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Execute the notebook cells to preprocess the data, engineer features, and train the models.

3. **Use Pre-Trained Models**:
   - Load the provided `.pkl` files to make predictions on new data.

4. **Modify the Pipeline**:
   - To experiment with different preprocessing techniques or model architectures, modify the pipeline structure in `starter.ipynb`.

---

## Built With

- [Python](https://www.python.org/) - Programming language used for all analysis and modeling.
- [scikit-learn](https://scikit-learn.org/) - Machine learning library for model training and evaluation.
- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis.
- [NumPy](https://numpy.org/) - Numerical computations.
- [nltk](https://www.nltk.org/) - Natural Language Processing library.
- [matplotlib](https://matplotlib.org/) - Visualization library.
- [seaborn](https://seaborn.pydata.org/) - Statistical data visualization.
- [Jupyter](https://jupyter.org/) - Interactive notebook environment.
- [pytest](https://pytest.org/) - Testing framework for verifying pipeline integrity.

All dependencies can be installed from the `requirements.txt` file.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/DSND-Pipelines-Project.git
   ```

2. Navigate into the project directory:

   ```bash
   cd DSND-Pipelines-Project
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

The `starter.ipynb` file contains all preprocessing, feature engineering, and model training steps. Open the notebook in Jupyter to execute the code step by step:

```bash
jupyter notebook starter.ipynb
```

### Accessing Pre-Trained Models

The repository includes pre-trained models:

- `gradient_boosting_model.pkl`
- `logistic_regression_model.pkl`
- `final_optimized_model.pkl`

To load these models for predictions, use the following code:

```python
import pickle

# Load Gradient Boosting Model
with open("gradient_boosting_model.pkl", "rb") as gb_file:
    gradient_boosting_model = pickle.load(gb_file)

# Load Logistic Regression Model
with open("logistic_regression_model.pkl", "rb") as lr_file:
    logistic_regression_model = pickle.load(lr_file)

# Example prediction
sample_data = [[35, 5, 0.8, 100, 10]]  # Replace with your preprocessed input
prediction = gradient_boosting_model.predict(sample_data)
print("Prediction:", prediction)
```

## Preprocessing Pipeline

### Data Preprocessing

The following steps ensure the input data is consistent with the training data:

1. **Combine Text Columns**:
   Combine `Title` and `Review Text` into a single column `Full Review`.

2. **Generate NLP Features**:
The pipeline includes the following advanced NLP features:
   - **Part-of-Speech (POS) Tagging**:
      - Identifies grammatical roles of words (e.g., noun, verb).
      - Used to filter specific word types for better sentiment analysis.

   - **Named Entity Recognition (NER)**:
      - Detects and categorizes entities such as brands, locations, or product names.

   - **Sentiment Analysis**:
      - Uses VADER sentiment analysis to assign compound sentiment scores to reviews.

   - **Word Counts**

3. **Encode Categorical Features**:
   Use `OneHotEncoder` to encode columns such as `Division Name`, `Department Name`, and `Class Name`.

4. **Scale Numerical Features**:
   Apply `StandardScaler` to numerical columns to standardize values.

### Example Preprocessing Code

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Example input data
example_data = pd.DataFrame({
    "Age": [35],
    "Positive Feedback Count": [5],
    "Sentiment Score": [0.8],
    "Word Count": [100],
    "Entity Count": [10],
    "Division Name": ["General"],
    "Department Name": ["Dresses"],
    "Class Name": ["Dresses"]
})

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
example_encoded = encoder.fit_transform(example_data[["Division Name", "Department Name", "Class Name"]])

# Scale numerical features
scaler = StandardScaler()
example_scaled = scaler.fit_transform(example_data[["Age", "Positive Feedback Count", "Sentiment Score", "Word Count", "Entity Count"]])

# Combine encoded and scaled features
import numpy as np
prepared_input = np.hstack((example_scaled, example_encoded))
```

## Results

### Gradient Boosting

- **Accuracy**: 88%
- **Class 0 Precision/Recall/F1**: 75.3% / 56.8% / 0.65
- **Class 1 Precision/Recall/F1**: 90.6% / 95.7% / 0.93

### Logistic Regression

- **Accuracy**: 87%
- **Class 0 Precision/Recall/F1**: 60.5% / 87% / 0.71
- **Class 1 Precision/Recall/F1**: 96.7% / 86.9% / 0.92

### Layered Inference

- **Accuracy**: 75.9%
- **Class 0 Precision/Recall/F1**: 35.7% / 35.7% / 0.36
- **Class 1 Precision/Recall/F1**: 85.2% / 85.2% / 0.85

## Feature Importance

The Gradient Boosting model provides insights into which features contribute most to predictions. Below is a visualization of feature importance:

```python
import matplotlib.pyplot as plt

# Example code for feature importance
importance = gradient_boosting_model.feature_importances_
features = X_train.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance for Gradient Boosting Model')
plt.show()
```

## License

This project is licensed under the terms specified in `LICENSE.txt`.
