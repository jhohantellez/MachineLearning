#  Machine Learning Web App - Salary Predictor
This project is an interactive web platform developed with **Flask** to explore **Supervised Learning** concepts. 
The core of the application is a **Multiple Linear Regression** model capable of predicting technology-sector salaries based on historical data patterns.

##  Interactive Use Cases
The app includes an interactive index highlighting four fundamental applications of Supervised Machine Learning:

1. **Salary Prediction** → Estimating income based on professional profiles *(Regression)*
2. **Credit Card Fraud Detection** → Identifying suspicious transactions *(Classification)*
3. **Virtual Assistants** → Natural Language Processing (NLP) and intent recognition
4. **Gmail Spam Filter** → Binary classification for email security

##  The Predictive Model
The system uses a **Multiple Linear Regression** model trained on a dataset of approximately **250,000 records** obtained from Kaggle.

###  Input Variables (X)
* `experience_years`: Total years of professional experience
* `skills_count`: Number of technical skills
* `certifications`: Number of professional certifications
###  Output Variable (y)
* `salary`: Predicted salary value (can represent monthly or annual income depending on dataset configuration)

## Tech Stack
* **Language:** Python 
* **Web Framework:** Flask
* **Data Science:** Pandas, Scikit-learn
* **Model Persistence:** Joblib
* **Visualization:** Matplotlib
* **Frontend:** HTML5, CSS3, Bootstrap

## Project Structure
```
MachineLearning/
│── app.py                # Main Flask application
│── templates/            # HTML files
│── static/               # CSS, JS, images
│── models/               # Trained ML models (.pkl)
│── requirements.txt      # Dependencies
```

##  Installation and Execution
1. Clone the repository:
```bash
git clone https://github.com/jhohantellez/MachineLearning.git
cd MachineLearning
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
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

##  Example Usage
1. Enter values such as years of experience, skills, and certifications
2. Submit the form
3. The system processes the input using the trained model
4. The predicted salary is displayed on screen
---
##  Authors
* Jhohan Téllez
* Andrey Lopez
* Felipe Navarro
---   
