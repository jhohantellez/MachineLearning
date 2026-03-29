from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

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

        prediction = model.predict([[experience, skills, certifications]])[0]

    return render_template("linear_regression/application.html", prediction=prediction)


#-------------------- MULTINOMIAL NAIVE BAYES --------------------
#-------------------- MULTINOMIAL MENU --------------------
@app.route("/multinomial")
def multinomial_menu():
    return render_template("multinomial_nb.html")


#-------------------- MULTINOMIAL CONCEPTS --------------------
@app.route("/multinomial/concepts")
def multinomial_concepts():
    return render_template("MultinomialNB/Basic_conceptsnb.html")


#-------------------- MULTINOMIAL APPLICATION --------------------
@app.route("/multinomial/application")
def multinomial_application():
    return render_template("MultinomialNB/Applicationnb.html")
#-------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)

print(app.url_map)