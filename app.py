from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        rf_model = joblib.load("model/rf_model.pkl")
        prediction = rf_model.predict([review])[0]
        if prediction == 1:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'
        return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
