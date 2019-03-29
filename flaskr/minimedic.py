import os
import hypothesis as H
import numpy as np
import pandas as pd
import pickle

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.run(debug=True)


@app.route('/_add_numbers')
def _add_numbers():
    print("hit")
    pelvic_incidence = request.args.get('pelvic_incidence', 0, type=float)
    pelvic_tilt = request.args.get('pelvic_tilt', 0, type=float)
    lumbar_lordosis_angle = request.args.get('lumbar_lordosis_angle', 0, type=float)
    sacral_slope = request.args.get('sacral_slope', 0, type=float)
    pelvic_radius = request.args.get('pelvic_radius', 0, type=float)
    degree_spondylolisthesis = request.args.get('degree_spondylolisthesis', 0, type=float)

    print("Looking for pickles...")
    h1 = pickle.load(open("h1.p", "rb"))
    h2 = pickle.load(open("h2.p", "rb"))
    h3 = pickle.load(open("h3.p", "rb"))
    print("Pickles found!")

    print("Predicting outcomes...")
    x_array = np.array([pelvic_incidence, pelvic_tilt,
                        lumbar_lordosis_angle, sacral_slope,
                        pelvic_radius, degree_spondylolisthesis])

    predict = [h1.predict(x_array), h2.predict(x_array), h3.predict(x_array)]
    print(predict)
    predicted_output = np.argmax(predict)
    print(predicted_output)
    options = ["Hernia", "Normal", "Spondylolisthesis"]
    outcome = options[predicted_output]

    return jsonify(result=outcome)


@app.route('/')
def index():
    return render_template('minimedic.html')
