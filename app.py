from flask import Flask, render_template, request
import LinearRegression
from LogisticRegression import pre_resultado
import joblib



app= Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    resultado = None
    if request.method == 'POST':
        try:

            datos_usuario = [
                float(request.form['edad']),
                float(request.form['ingreso_mensual']),
                float(request.form['visitas_web_mes']),
                float(request.form['tiempo_sitio_min']),
                float(request.form['compras_previas']),
                float(request.form['descuento_usado'])
            ]
             
            res_binario = pre_resultado(datos_usuario)
            
            resultado = "EL CLIENTE COMPRARÁ (1)" if res_binario == 1 else "EL CLIENTE NO COMPRARÁ (0)"
            
        except Exception as e:
            resultado = f"Error en la predicción: {e}"

    return render_template('logisticRegression.html', prediction_text=resultado)

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

#-------------------------------------------------------------------------

@app.route("/linearregression/concepts")
def concepts():
    return render_template("linear_regression/basic_concepts.html")

@app.route("/linearregression/application", methods=["GET", "POST"])
def linear_regression():
    prediction = None

    if request.method == "POST":
        experience = float(request.form["experience"])
        skills = float(request.form["skills"])
        certifications = float(request.form["certifications"])


        prediction = model.predict([[experience, skills, certifications]])[0]

    return render_template("linear_regression/application.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


print(app.url_map)
