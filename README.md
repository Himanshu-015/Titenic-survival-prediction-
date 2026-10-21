# Titanic Survival Prediction 🚢

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using Machine Learning. It uses a Random Forest Classifier and a Streamlit web application to make predictions from uploaded Excel files.

## Objectives
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Handle missing values and outliers.
- Train and evaluate a Random Forest classification model.
- Improve model performance using cross-validation and GridSearchCV.
- Deploy the model using Streamlit.
- Generate survival predictions and downloadable results.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Random Forest Classifier
- GridSearchCV and Cross-Validation
- Joblib
- Streamlit
- Jupyter Notebook

## Project Files

- `Untitled3.ipynb` — Data analysis, preprocessing, model training, and evaluation.
- `app.py` — Streamlit web application.
- `final_classification_model.joblib` — Trained classification model.
- `preprocessor_pipeline.joblib` — Saved preprocessing pipeline.
- `titanic.csv` — Dataset, if included in the repository.
- `requirements.txt` — Required Python libraries.

*Note: The filenames above should match the actual files uploaded to the repository.*

## Key Features
- Data exploration and visualization.
- Missing-value imputation and outlier handling.
- Numerical and categorical feature preprocessing.
- Random Forest model training.
- Hyperparameter tuning with GridSearchCV.
- Excel file upload for batch predictions.
- Survival status and survival probability.
- Downloadable prediction results in CSV format.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Himanshu-015/Titenic-survival-prediction-.git
cd Titenic-survival-prediction-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib openpyxl
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## How to Use
1. Start the Streamlit application.
2. Upload an Excel file containing the required passenger features.
3. Review the uploaded data.
4. Generate survival predictions.
5. View the predicted survival status and probability.
6. Download the results as a CSV file.

## Machine Learning Workflow
1. Data loading and inspection.
2. Exploratory Data Analysis.
3. Data cleaning and preprocessing.
4. Train-test splitting.
5. Feature transformation.
6. Random Forest model training.
7. Cross-validation and hyperparameter tuning.
8. Model saving with Joblib.
9. Deployment with Streamlit.

## Learning Outcomes
This project demonstrates an end-to-end Machine Learning workflow, from data analysis and model development to deployment through an interactive web application.

## Limitations
Predictions depend on the quality of the input data and the trained model. The results are educational predictions and should not be treated as historically definitive for individual passengers.

## Author
**Himanshu**  
M.Tech — Computer Science and Engineering

## License
This project is intended for educational and learning purposes.
