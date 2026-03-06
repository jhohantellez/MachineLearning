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
print(app.url_map)