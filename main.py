import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.compose import make_column_transformer,ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import r2_score
app = Flask(__name__, template_folder=r"C:\Users\HP\PycharmProjects\pythonProject")
data = pd.read_csv("cleaned_data.csv")

import pickle
model=pickle.load(open("RidgeModel.pkl","rb"))
locations = sorted(data['location'].unique())
# pickle="E:\AI ML DATA SET\banglore house price pred\RidgeModel.pkl"
@app.route('/')
def index():

    return render_template('index.html', locations=locations)


@app.route('/', methods=['POST'])
def predict():
    location=request.form["location"]
    bath=request.form["bath"]
    bhk=request.form["bhk"]
    sqft=request.form["total_sqft"]
    data=pd.DataFrame({"location":[location],"total_sqft":[sqft],"bath":[bath],"bhk":[bhk]})
    res=model.predict(data)
    return render_template('index.html',locations=locations,result=(res[0]*1e5))
if __name__ == "__main__":
    app.run(debug=True, port=9119)
