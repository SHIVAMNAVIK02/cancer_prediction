from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load models
regressor_model = pickle.load(open('models/model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        Age=int(request.form.get('Age'))
        Gender=int(request.form.get('Gender'))
        BMI=float(request.form.get('BMI'))
        Smoking=int(request.form.get('Smoking'))
        GeneticRisk=int(request.form.get('GeneticRisk'))
        PhysicalActivity=float(request.form.get('PhysicalActivity'))
        AlcoholIntake=float(request.form.get('AlcoholIntake'))
        CancerHistory=int(request.form.get('CancerHistory'))

        data = standard_scaler.transform([[Age,Gender,BMI,Smoking,GeneticRisk,PhysicalActivity,AlcoholIntake,CancerHistory]])
        result = regressor_model.predict(data)

        return render_template("home.html",results=result[0])


    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
