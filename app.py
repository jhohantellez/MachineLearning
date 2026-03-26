from flask import Flask, render_template, request
import joblib


app= Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def Home():
    return render_template('home.html')

@app.route("/usecases")
def UseCase():
    return render_template('index.html')

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

@app.route("/linearregresion")
def linearregression():
    return render_template('index.html')

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


#-------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


print(app.url_map)
