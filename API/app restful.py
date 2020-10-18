from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import logging
import pickle
import os

port = int(os.getenv("PORT", 9099))

app = Flask(__name__)
api = Api(app)

items = []
logging.basicConfig(level=logging.DEBUG)

scaler = pickle.load(open("customer_kmeans_scaler.pickle","rb"))
kmeans = pickle.load(open("customer_kmeans_segmentation.pickle","rb"))

def segment_label(cluster_number):
    if cluster_number == 0:
        return "Fan"
    elif cluster_number == 1:
        return "Roamer"
    elif cluster_number == 2:
        return "Supporter"
    else :
        return "Alienated"

class Segmentation(Resource):

    def post(self):
        data = request.get_json()
        sat_score = data['satisfaction']
        spend_amt = data['spend']
        visits = data['visits']
        msg = 'Satisfaction : {satisfaction} , spend : {spend} , visits {visits}'.format(satisfaction=sat_score, spend=spend_amt, visits=visits)
        logging.debug(msg)

        prediction_data = [[sat_score, spend_amt, visits]]
        prediction_data_scaled = scaler.transform(prediction_data)

        prediction = kmeans.predict(prediction_data_scaled)
        pred_segment = segment_label(prediction[0])
        segment = {
            'segment': pred_segment
        }
        return jsonify(segment)



api.add_resource(Segmentation, '/segmentation')

#host needs to be 0.0.0.0 to make it visible externally
app.run(host='0.0.0.0',port=port)
