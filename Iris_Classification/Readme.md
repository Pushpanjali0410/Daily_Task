# Iris Flower Classification Using Machine Learning

## Project Overview

This project implements a Machine Learning Classification Model using the famous Iris Dataset available in Scikit-Learn. The model classifies iris flowers into three species:

* Setosa
* Versicolor
* Virginica

The project includes:

* Exploratory Data Analysis (EDA)
* Data Visualization
* Model Training
* Model Evaluation
* Model Saving using Pickle
* Prediction on New Samples

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Pickle

---

## Project Structure

```text
Iris-Classification-Project/
│
├── models/
│   └── iris_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── correlation_heatmap.png
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Information

The Iris Dataset contains 150 samples with 4 numerical features:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

Target Classes:

* Setosa
* Versicolor
* Virginica

---

## Exploratory Data Analysis

Performed:

* Dataset Inspection
* Missing Value Analysis
* Class Distribution Analysis
* Correlation Analysis
* Visualization using Heatmaps

---

## Machine Learning Model

Algorithm Used:

* Random Forest Classifier

Reasons:

* High Accuracy
* Robust to Overfitting
* Works well on small datasets

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score

Expected Accuracy:

```text
95% - 100%
```

---

## Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Iris-Classification-Project.git
cd Iris-Classification-Project
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python src/train.py
```

This will:

* Train the model
* Generate evaluation metrics
* Save visualizations
* Save model as iris_model.pkl

---

## Run Prediction

```bash
python src/predict.py
```

Example Output

```text
Predicted Flower:
setosa
```

---

## Model Saving

The trained model is saved in:

```text
models/iris_model.pkl
```

---

## Future Improvements

* Streamlit Web Application
* Hyperparameter Tuning
* Model Comparison
* Deployment on Cloud

---

## Author

Pushpanjali

Machine Learning Internship Project
