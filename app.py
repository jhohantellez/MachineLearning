from flask import Flask, render_template, request




app= Flask(__name__)

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

#-------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


print(app.url_map)
