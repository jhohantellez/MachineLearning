from flask import Flask, render_template, request
import joblib
from train_multinominal import get_trained_model
import Clustering

app = Flask(__name__)

from logisticmodel import pre_resultado

model_lr = joblib.load("model.pkl")
model_lr = joblib.load("model.pkl")
model_nb, vectorizer_nb, nb_metrics = get_trained_model()

#-------------------- HOME --------------------
@app.route("/")
def home():
    return render_template('home.html')

#-------------------- USE CASES --------------------
@app.route("/usecases")
def usecases():
    return render_template('usecases.html')

@app.route("/UseCase/1")
def usecase1():
    return render_template('use_cases/use_case1.html')

@app.route("/UseCase/2")
def usecase2():
    return render_template("use_cases/use_case2.html")

@app.route("/UseCase/3")
def usecase3():
    return render_template("use_cases/use_case3.html")

@app.route("/UseCase/4")
def usecase4():
    return render_template("use_cases/use_case4.html")

#-------------------- LINEAR REGRESSION --------------------
@app.route("/linearregression")
def linearregression():
    return render_template('linearregression.html')

@app.route("/linearregression/concepts")
def concepts():
    return render_template("linear_regression/basic_concepts.html")

@app.route("/linearregression/application", methods=["GET", "POST"])
def application():
    prediction = None
    if request.method == "POST":
        experience = float(request.form["experience"])
        skills = float(request.form["skills"])
        certifications = float(request.form["certifications"])
        if model_lr:
            prediction = model_lr.predict([[experience, skills, certifications]])[0]
    return render_template("linear_regression/application.html", prediction=prediction)

#------------------------LOGISTIC REGRESSION-----------------------------------------------
@app.route("/logisticregression")
def logisticregression():
    return render_template("logisticregression.html")

@app.route("/logisticregression/concepts")
def conceptslogistic():
    return render_template("logistic_regression/concepts_logistic.html")

@app.route('/logisticregression/application', methods=['GET', 'POST'])
def applicationlogistic():
    prediction = None
    probability = None

    if request.method == 'POST':
        data_input = {
            "Age": float(request.form['Age']),
            "Income": float(request.form['Income']),
            "Credit_Score": float(request.form['Credit_Score']),
            "Loan_Amount": float(request.form['Loan_Amount']),
            "Loan_Term": float(request.form['Loan_Term']),
            "Employment_Status": request.form['Employment_Status']
        }

        result, prob = pre_resultado(data_input)

        if result == 1:
            prediction = "Loan Approved "
        else:
            prediction = "Loan Not Approved "

        probability = f"{prob:.2f}"

    return render_template('logistic_regression/application_logistic.html', prediction=prediction, probability=probability)


#-------------------- MULTINOMIAL NAIVE BAYES --------------------
@app.route("/multinomial")
def multinomial_menu():
    return render_template("multinomial_nb.html")

@app.route("/multinomial/concepts")
def multinomial_concepts():
    return render_template("MultinomialNB/Basic_conceptsnb.html")

@app.route("/multinomial/application", methods=["GET", "POST"])
def multinomial_application():
    prediction = None
    original_text = None
    
    if request.method == "POST":
        message = request.form['message']
        original_text = message
        
        vect = vectorizer_nb.transform([message])
        prediction_num = model_nb.predict(vect)[0]
        
        prediction = "SPAM DETECTED! " if prediction_num == 1 else "Legitimate Message (HAM)"
    
    return render_template("MultinomialNB/Applicationnb.html", metrics=nb_metrics,  prediction=prediction, original_text=original_text)


#-------------------- CLUSTERING --------------------
@app.route("/clustering")
def clustering():
    return render_template("clustering.html")

@app.route("/clustering/concepts")
def clustering_concepts():
    return render_template("clustering/concepts_clustering.html")

@app.route("/clustering/application")
def clustering_application():
    k = request.args.get("k", default=3, type=int)

    info = Clustering.AppClusteringKmeans(k)
    centers = info["centers"]
    centers = [[round(value, 2) for value in center] for center in centers]
    summary = info["summary"]
    results = info["results"]

    return render_template(
        "clustering/application_clustering.html",
        centers=enumerate(centers),
        summary=summary,
        results=results,
        k=k
    )
#-------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)