from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("DBS_SGD.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    q = float(request.form["q"])
    data = pd.DataFrame({"SGD": [q]})
    r = round(float(model.predict(data)[0][0]), 2)
    return render_template("prediction.html", r=r)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)