from flask import Flask, render_template, request
import LinearRegression
from LogisticRegression import pre_resultado
from flask import Flask, render_template

app= Flask(__name__)

#create new route
@app.route('/')

#defined function
def home ():
    return 'hello flask'

@app.route("/FirstPage")
def firstPage():
    return render_template('index.html')

@app.route('/LinearRegression/',methods=["GET","POST"])
def calculateGrade():
    calculateResult= None
    if request.method == "POST":
        hours= float(request.form["hours"])
        calculateResult= LinearRegression.calculateGrade(hours)
    return render_template("linearRegressionGrades.html", result= calculateResult)

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

if __name__ == "__main__":
    app.run(debug=True)


print(app.url_map)
