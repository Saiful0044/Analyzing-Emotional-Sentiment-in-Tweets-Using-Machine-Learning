from flask import Flask,render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import joblib

"""
mlflow.set_tracking_uri("https://dagshub.com/Saiful0044/Analyzing-Emotional-Sentiment-in-Tweets-Using-Machine-Learning.mlflow")

dagshub.init(repo_owner='Saiful0044', repo_name='Analyzing-Emotional-Sentiment-in-Tweets-Using-Machine-Learning', mlflow=True)
# load model from registry
model_name = 'my_model'
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

"""

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow 
    features = vectorizer.transform([text])

    # predict
    result = model.predict(features)
    
    return render_template('index.html', result=result[0])
app.run(debug=True)