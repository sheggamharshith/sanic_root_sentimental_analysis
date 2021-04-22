# sanic import
from flair.data import Sentence
from flair.models import TextClassifier
from sanic import Sanic
from sanic.response import json
from sanic.log import logger
from sanic_cors import CORS, cross_origin
from itertools import count

counter = 0

# falir import


app = Sanic(__name__)
CORS(app)

# loading the sentiment analyzier
classifier = TextClassifier.load('en-sentiment')

# global
flip_value = {"NEGATIVE": "POSITIVE", "POSITIVE": "NEGATIVE"}


@app.route("/")
async def test(request):
    return json({"detail": "Welcome to sentimental analysis API v1"})


@app.route('/v1/predict/', methods=['POST'])
@cross_origin(app)
async def sentimentAnalysis(request):

    response_data = {"sentence": None, "request_id": None,
                     "NEGATIVE": 0, "POSITIVE": 0}
    try:
        input_body_json = request.json
        sentence = Sentence(input_body_json['sentence'])
        classifier.predict(sentence)
        logger.info(f'Sentiment: {sentence.labels}')
        label = sentence.labels[0]
        labscore = (label.score)*100
        response_data[f'{label.value}'] = labscore
        response_data[f"{flip_value[f'{label.value}']}"] = 100-labscore
        response_data['sentence'] = request.json['sentence']
        response_data['request_id'] = request.json['request_id']
        return json(body=response_data, status=200)
    except Exception as e:
        logger.error(f"{e}")
        return json(body={"detail": f"{e}"}, status=404)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, access_log=True)
