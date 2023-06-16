import os
import inspect
import logging

app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

import tensorflow as tf

tf.gfile = tf.io.gfile
logging.basicConfig(level=logging.INFO)
import traceback  # noqa
from dotenv import load_dotenv

from django.shortcuts import render
from django.views import View
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .apps import DetectorConfig


NUMBER_WORDS = 100


class FakeNewsView(View):
    template_name = 'fake_news.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        input_text = request.POST.get('input_text', '')
        tokenizer = DetectorConfig.token_model
        lstm_model = DetectorConfig.lstm_model
        text_seq = tokenizer.texts_to_sequences([input_text])
        test_input = pad_sequences(text_seq, maxlen=NUMBER_WORDS)

        output = lstm_model.predict(test_input)

        if output >= 0.5:
            context = {
                'input_text': input_text,
                'is_genuine': 1 if output[0][0] >= 0.5 else 0,
                'probability': round(output[0][0] * 100, 2)
            }
        else:
            context = {
                'input_text': input_text
            }

        return render(request, self.template_name, context)
