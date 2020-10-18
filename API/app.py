from flask import Flask, request, jsonify
import logging
import pickle
import os

logging.basicConfig(level=logging.DEBUG)
port = int(os.getenv("PORT", 9099))
app = Flask(__name__)
items = []

kmeans_scaler = pickle.load(open("customer_kmeans_scaler.pickle","rb"))
kmeans = pickle.load(open("customer_kmeans_segmentation.pickle","rb"))
car_purchase_scaler = pickle.load(open("car_purchase_scaler.pickle","rb"))
car_purchase_predictor = pickle.load(open("car_purchase_predictor.pickle","rb"))

def segment_label(cluster_number):
    if cluster_number == 0:
        return "Fan"
    elif cluster_number == 1:
        return "Roamer"
    elif cluster_number == 2:
        return "Supporter"
    else :
        return "Alienated"

@app.route("/segmentation", methods=["POST"])
def segment():
    data = request.get_json()
    sat_score = data['satisfaction']
    spend_amt = data['spend']
    visits = data['visits']
    msg = 'Satisfaction : {satisfaction} , spend : {spend} , visits {visits}'.format(satisfaction=sat_score, spend=spend_amt, visits=visits)
    logging.debug(msg)

    prediction_data = [[sat_score, spend_amt, visits]]
    prediction_data_scaled = kmeans_scaler.transform(prediction_data)

    prediction = kmeans.predict(prediction_data_scaled)
    pred_segment = segment_label(prediction[0])
    segment = {
        'segment': pred_segment
    }
    msg = 'Segment : {pred_segment}'.format(pred_segment=pred_segment)
    logging.debug(msg)
    return jsonify(segment)

@app.route("/prediction", methods=["POST"])
def predict():
    data = request.get_json()
    sat_score = data['satisfaction']
    vehicle_age = data['vehicleAge']
    vehicle_value = data['value']
    msg = 'Satisfaction : {satisfaction} , age : {age} , value {value}'.format(satisfaction=sat_score, age=vehicle_age, value=vehicle_value)
    logging.debug(msg)

    prediction_data = [[sat_score, vehicle_age, vehicle_value]]
    prediction_data_scaled = car_purchase_scaler.transform(prediction_data)

    prediction = car_purchase_predictor.predict(prediction_data_scaled)
    pred_value = int(prediction[0])
    response = {
        'prediction': pred_value
    }
    msg = 'Prediction : {pred}'.format(pred=pred_value)
    logging.debug(msg)
    return jsonify(response)



#host needs to be 0.0.0.0 to make it visible externally
app.run(host='0.0.0.0',port=port)

