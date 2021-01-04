from aiohttp.web import json_response
from json.decoder import JSONDecodeError
import pandas as pd

import os
import logging
from datetime import datetime

from libs.utils.errors import ApiBadRequest
from libs.models.CRNNModel import CRNNModel
from config import pretrained_model, n_epochs, csv_path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class RouterHandler(object):
    def __init__(self, loop):
        self._loop = loop
        self.model = CRNNModel(model_path=pretrained_model, initial_state=False)

    async def train(self, request):
        try:
            epochs = int(request.rel_url.query['epochs'])
        except Exception as err:
            print(err)
            epochs = n_epochs

        start = datetime.now()
        self.model.build_model('train')
        self.model.fit(epochs=epochs)
        end = datetime.now()
        return json_response({
            "status": "Success",
            "time": str(end - start)
        })

    async def evaluation(self, request):
        start = datetime.now()
        body = await decode_request(request)
        able_fields = ['filename']
        body = filter_fields(able_fields, body)

        file = os.path.join(csv_path, 'test.csv')
        if body.get('filename') is not None:
            file = body.get('filename')

        try:
            data = pd.read_csv(file, sep=';')
        except FileNotFoundError as err:
            logging.exception(err)
            return json_response({
                "status": "Fail",
                "detail": "File not found"
            })

        paths = body.get('paths')
        labels = body.get('labels')

        if paths is None or labels is None:
            paths = data['Image'].values.tolist()
            labels = data['Label'].values.tolist()

        accuracy, letter_accuracy = self.model.evaluate(paths, labels)
        end = datetime.now()
        time = end - start

        return json_response({
            "status": "Success",
            "accuracy": accuracy,
            "letter_accuracy": letter_accuracy,
            "time": time.total_seconds()
        })

    async def prediction(self, request):
        start = datetime.now()
        body = await decode_request(request)
        required_fields = ['image']
        validate_fields(required_fields, body)

        img = body.get('image')
        predicted = self.model.predict(img)
        end = datetime.now()
        time = end - start

        if predicted is None:
            return json_response({
                "status": "Fail",
                "detail": "Image not found"
            })

        return json_response({
            "status": "Success",
            "predicted": predicted,
            "time": time.total_seconds()
        })


async def decode_request(request):
    try:
        return await request.json()
    except JSONDecodeError:
        raise ApiBadRequest('Improper JSON format')


def validate_fields(required_fields, body):
    for field in required_fields:
        if body.get(field) is None:
            raise ApiBadRequest("'{}' parameter is required".format(field))


def filter_fields(able_fields, body):
    result = {}
    for field in able_fields:
        if body.get(field):
            result[field] = body.get(field)
    return result
