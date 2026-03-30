from flask import Flask, render_template, request
import joblib
from logisticmodel import pre_resultado

app= Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def Home():
    return render_template('home.html')

@app.route("/usecases")
def UseCase():
    return render_template('usecases.html')

#-----------------Use Cases-------------------------------------------
@app.route("/UseCase/1")
def UseCase1():
    return render_template('use_cases/use_case1.html')

@app.route("/UseCase/2")
def UseCase2():
    return render_template("use_cases/use_case2.html")

@app.route("/UseCase/3")
def UseCase3():
    return render_template("use_cases/use_case3.html")

@app.route("/UseCase/4")
def UseCase4():
    return render_template("use_cases/use_case4.html")

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


        prediction = model.predict([[experience, skills, certifications]])[0]

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




#-------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


print(app.url_map)
