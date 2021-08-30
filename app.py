from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)
model = pickle.load(open('ridge_reg.sav', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == "POST":
        Process_temperature = float(request.form['Process_temperature'])
        Rotational_speed= int(request.form['Rotational_speed'])
        Torque= float(request.form['Torque'])
        Tool_wear = int(request.form['Tool_wear'])
        TWF = int(request.form['TWF'])
        HDF = int(request.form['HDF'])
        PWF = int(request.form['PWF'])
        OSF = int(request.form['OSF'])
        RNF = int(request.form['RNF'])

        scaler = StandardScaler()

        input = scaler.fit_transform([[Process_temperature,Rotational_speed,Torque,Tool_wear,TWF,HDF,PWF,OSF,RNF]])

        prediction = model.predict(input)

        return render_template('home.html', prediction_text="Your Air Temperature is {}".format(prediction))
    else:
        return render_template("home.html")




if __name__ == '__main__':
    app.run(debug=True)
