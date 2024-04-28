from flask import Flask
from flask import request
from flask import jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import csv
import io
from experiment import saved_filename
from data_check import DataCheck
import pandas as pd
import json

app = Flask(__name__)
auth = HTTPBasicAuth()

credential_path = "E:/Documents/Python Documents/HeartDisease/credentials.json"
with open(credential_path,'rb') as json_file:
    credentials = json.load(json_file)
users = credentials.get("user")


@auth.verify_password
def verify_password(username, password):
    print("USERNAME:", username)
    print("CORRECT PSW:",users.get(username))
    if username in users:
        username_hash = generate_password_hash(users.get(username))
        if check_password_hash(username_hash, password):
            return username


@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    # access data in bytestring (assume client send .csv file)
    if request.content_type != 'text/csv':
        warning = f"Incorrect data type: must be csv file, you sent {request.content_type}"
        return warning
    predictors_string = request.get_data(as_text=True)
    check_instance = DataCheck(predictors_string)
    check_result = check_instance.attribute_check()
    if not check_result == "passed":
        return check_result
    # print("FILE RECEIVED: ", predictors_string)
    names = ('age','sex','cp','trestbps','chol','fbs',
             'restecg', 'thalach', 'exang', 'oldpeak', 'slop', 'ca'
             ,'thal')
    predictors = pd.read_csv(io.StringIO(predictors_string), names=names)
    # perform basic data check

    # drop irrelevant attributes
    dropped_cols = ['sex', 'trestbps', 'fbs',]
    # predictors.drop_features(dropped_cols)
    # predictors.drop(labels=dropped_cols, axis=1)
    print("DATAFRAME: ", predictors)
    print("SIZE OF DATA: ",predictors.shape)
    predictors = predictors.drop(dropped_cols, axis=1)
    # load saved model
    file_path = 'E:/Documents/Python Documents/HeartDisease/'+saved_filename
    loaded_model = pickle.load(open(file_path,'rb'))
    results = loaded_model.predict(predictors)
    # jsonalise results:
    # results = jsonify(results)
    results = results.tolist()
    print("THE RESULT IS: ", results)
    return results


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=100, use_reloader=True)
