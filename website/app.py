from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

def load_label_encoders():
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        model, scaler, label_encoders = pickle.load(file)
    return model, scaler, label_encoders

def prediction(lst):
    model, scaler, label_encoders = load_label_encoders()

    lst_with_id = [0] + lst  

    lst_scaled = scaler.transform([lst_with_id])  

    pred_value = model.predict(lst_scaled)
    return pred_value

@app.route('/', methods=['POST', 'GET'])
def index():
    pred_value = 0
    if request.method == 'POST':
        meals_served = request.form['meals_served']
        kitchen_staff = request.form['kitchen_staff']
        temperature_C = request.form['temperature_C']
        humidity_percent = request.form['humidity_percent']
        day_of_week = request.form['day_of_week']
        special_event = request.form.get('special_event', 'no')
        past_waste_kg = request.form['past_waste_kg']
        staff_experience = request.form['staff_experience']
        waste_category = request.form['waste_category']

        feature_list = [
            float(meals_served),          
            int(kitchen_staff),            
            float(temperature_C),           
            float(humidity_percent),       
            int(day_of_week),              
            1 if special_event == 'yes' else 0, 
            float(past_waste_kg),           
            int(staff_experience),         
            ['low', 'medium', 'high'].index(waste_category) 
        ]

        pred_value = prediction(feature_list)
        pred_value = np.round(pred_value[0], 2) 
    return render_template('index.html', pred_value=pred_value)

if __name__ == '__main__':
    app.run(debug=True)
