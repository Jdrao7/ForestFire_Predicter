# Algerian Forest Fire Prediction 🔥🌲

This project is a complete end-to-end machine learning endeavor focused on predicting the **Fire Weather Index (FWI)** in the Algerian forest region. The goal is to build a regression model that accurately forecasts fire danger based on meteorological data. The project covers the entire ML lifecycle, from data cleaning and exploratory data analysis (EDA) to model training, evaluation, and deployment.

## ✨ Key Features

-   **🧹 Data Cleaning & Preprocessing**: Handles missing values, corrects data type inconsistencies, and prepares the dataset for analysis.
-   **📊 Exploratory Data Analysis (EDA)**: Utilizes libraries like Pandas Profiling and various visualizations (e.g., histograms, heatmaps) to uncover insights, correlations, and patterns in the data.
-   **⚙️ Feature Engineering & Selection**: Derives new features and selects the most impactful ones for model training.
-   **🧠 Model Training**: Implements a **Ridge Regression** model to predict the Fire Weather Index.
-   **✔️ Model Evaluation**: Assesses model performance using metrics like **R-squared** and **Adjusted R-squared**.
-   **📦 Model Persistence**: Saves the trained model and the data preprocessor (StandardScaler) as pickle files for easy deployment.

## ⚙️ Technology Stack

-   **Data Analysis & Manipulation**: Pandas, NumPy
-   **Data Visualization**: Matplotlib, Seaborn
-   **Machine Learning**: Scikit-learn
-   **Development Environment**: Jupyter Notebook

## 📂 Repository Structure
├── notebooks/
│   ├── 1. EDA Algerian Forest.ipynb
│   └── 2. Model Training.ipynb
├── Algerian_forest_fires_dataset_UPDATE.csv
└── README.md
-   **`notebooks/`**: Contains the Jupyter Notebooks detailing the step-by-step process.
    -   **`1. EDA Algerian Forest.ipynb`**: The notebook for data cleaning, preprocessing, and exploratory data analysis.
    -   **`2. Model Training.ipynb`**: The notebook for feature engineering, model building, training, and evaluation.
-   **`Algerian_forest_fires_dataset_UPDATE.csv`**: The primary dataset used for this project.

## 🚀 How to Run this Project

### Prerequisites

-   Python 3.x
-   Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Jdrao7/ML---Algerian-Forest.git](https://github.com/Jdrao7/ML---Algerian-Forest.git)
    cd ML---Algerian-Forest
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

4.  **Launch Jupyter Notebook:**
    ```sh
    jupyter notebook
    ```

5.  **Run the notebooks:**
    -   Open and run the notebooks in the `notebooks/` directory in sequential order, starting with `1. EDA Algerian Forest.ipynb` and then `2. Model Training.ipynb`.

## 📈 Results

The final Ridge Regression model achieved the following performance:
-   **R-squared**: [Insert R-squared value from your notebook]
-   **Adjusted R-squared**: [Insert Adjusted R-squared value from your notebook]

This indicates that the model is able to effectively explain the variance in the Fire Weather Index based on the provided features.

## 🙏 Acknowledgements

This project is based on the Algerian Forest Fires Dataset from the UCI Machine Learning Repository.
