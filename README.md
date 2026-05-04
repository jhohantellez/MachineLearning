# Machine Learning Web Application

This project consists of an interactive web application developed using Flask, whose objective is to demonstrate the practical implementation of **supervised and unsupervised machine learning algorithms** within a structured web environment.

The application integrates theoretical concepts, manual algorithm simulation, and real-time execution of machine learning models, facilitating both learning and experimentation.

---

## Project Overview

The system implements different Machine Learning techniques within a single platform:

* **Linear Regression** → Salary prediction
* **Logistic Regression** → Classification problems
* **Multinomial Naive Bayes** → Text classification (spam detection)
* **K-Means Clustering** → Customer segmentation

Each of these models is integrated through specific Flask routes, enabling interactive use through the web interface.

---

## System Architecture

The application follows a modular architecture based on Flask:

* `app.py` → Main controller and route handler
* `templates/` → Views organized by machine learning model
* `static/` → Static files (CSS, images)
* `models/` → Trained machine learning models

### Application Flow

1. The user inputs data through web forms
2. Flask processes the request via a specific route
3. The corresponding model is executed
4. A result is generated (prediction or clustering)
5. The result is dynamically displayed in the interface

---

## Machine Learning Modules

### 1. Linear Regression

* Supervised learning model
* Used for salary prediction
* Based on structured numerical data
* Implemented using Scikit-learn

---

### 2. Logistic Regression

* Binary classification model
* Used in decision-making systems (e.g., loan approval)

---

### 3. Multinomial Naive Bayes

* Text classification model
* Applied to spam detection
* Based on probabilistic word distributions

---

### 4. K-Means Clustering

#### Objective

To segment customers based on financial behavior using:

* `Credit_Limit`
* `Total_Trans_Amt`

#### Dataset

* Source: Kaggle (BankChurners dataset)
* Contains information about customer financial behavior

---

## K-Means Implementation

The clustering module includes both theoretical explanation and practical implementation.

### Data Preparation Process

1. Dataset loading from CSV file
2. Selection of relevant features
3. Data cleaning (handling missing values)
4. Standardization using `StandardScaler`
5. Conversion to numerical format

---

### Model Training

* Algorithm: K-Means

* Parameters used:

  * `n_clusters = 4`
  * `n_init = 10`
  * `random_state` for reproducibility

* Main method:

```python
fit_predict()
```

---

### Generated Results

* Cluster assignment for each record
* Centroid calculation
* Cluster distribution summary
* Representative sample per cluster
* Graphical visualization

---

## Cluster Analysis

### Distribution

* Cluster 0 → Low credit and low activity
* Cluster 1 → High activity users
* Cluster 2 → High credit and low usage
* Cluster 3 → Moderate behavior

---

## Visualization

The system generates scatter plots using Matplotlib:

* Each point represents a customer
* Colors indicate cluster membership
* Centroids are visually highlighted

The plots are converted to **base64 format**, allowing direct integration into the web interface without storing image files.

---

## Frontend Structure

The application is organized into different sections:

* **Home:** Introduction to Machine Learning
* **Concepts:** Theoretical explanations
* **Manual Exercise:** Step-by-step algorithm simulation
* **Interactive Application:** Real-time execution

This structure combines theory, guided practice, and interactive experimentation.

---

## Project Structure

```
MachineLearning/
│── app.py
│── templates/
│   ├── clustering/
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── MultinomialNB/
│   └── Use_Cases.html
│── static/
│   ├── css/
│   ├── images/
│── models/
│── requirements.txt
```

---

## Installation and Execution

1. Clone the repository:

```bash
git clone https://github.com/jhohantellez/MachineLearning.git
cd MachineLearning
```

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. Open in your browser:

```
http://127.0.0.1:5000/
```

---

## Usage

1. Navigate through the Machine Learning modules
2. Input data into the forms
3. Execute the selected model
4. Analyze the generated results

---

## Key Features

* Integration of multiple Machine Learning algorithms
* Real-time execution
* Data visualization
* Modular architecture based on Flask
* Educational and practical approach

---

## Deployment

The application is deployed on Render:

https://machinelearning-ldtz.onrender.com

---

## Authors

* Jhohan Téllez
* Andrey López
* Felipe Navarro
