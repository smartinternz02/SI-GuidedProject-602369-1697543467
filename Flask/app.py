from flask import Flask, render_template, request
import pickle
import numpy as np

modelfile = "C:\\Users\\Naveen\\OneDrive\\Desktop\\wholesale_customer_segmentation_ML\\models\\final_prediction.pickle"

model = pickle.load(open(modelfile, 'rb')) 

# Remove duplicate import of Flask and requestTraining_models
app = Flask(__name__) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/value')
def value():
    return render_template('value.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        #Get form data as strings
        Channel = request.form.get("Channel")
        Region = request.form.get('Region')
        Fresh = request.form.get('Fresh')
        Milk = request.form.get('Milk')
        Grocery = request.form.get('Grocery')
        Frozen = request.form.get('Frozen')
        Detergents_Paper = request.form.get('Detergent_paper')  # Corrected field name
        Delicassen = request.form.get('Delicassen')

        # Check if any of the form fields are None or empty
        if None in [ Grocery,Frozen, Detergents_Paper, Delicassen]:
            result = "Please fill out all form fields with valid numbers."
        else:
            # Convert form data to float
            total = [ float(Grocery), float(Frozen), float(Detergents_Paper), float(Delicassen)]
            total = np.array(total).reshape(1, -1)  # Reshape for model prediction

            prediction = model.predict(total)

            if prediction == 0:
                result = "Customer Belongs to Cluster Label 0"
            elif prediction == 1:
                result = "Customer Belongs to Cluster Label 1"
            elif prediction == 2:
                result = "Customer Belongs to Cluster Label 2"
            else:
                result = "Customer Belongs to Cluster Label 3"

    return render_template('predict.html', predict=result)



if __name__ == '__main__':
    app.run(debug=True)
