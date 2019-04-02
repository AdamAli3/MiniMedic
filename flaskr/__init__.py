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

    hernia_description = "A herniated disk occurs when an intervertebral disc ruptures due to excessive strain or injury, most commonly in the lower spine. If the herniated disc is pressing up against a nerve, the patient may experience pain, numbness and weakness. Common risk factors include excessive strain or injury due to excess body weight or repetitive lifting, pulling and twisting. Herniated disks can also be inherited."
    hernia_treatment = "Herniated discs are frequently treated with nonsteroidal anti-inflammatory medication. Doctors may prescribe bedrest or a painless activity level. Physical therapy including pelvic traction, gentle massages, ice and heat therapy, EMS and stretching exercises are often very beneficial. In extreme cases, surgery may be recommended if physical therapy and medication if physical therapy and medication do not ease or eliminate the pain."

    spond_description = "Spondylolisthesis is when a vertebra slips out of alignment and onto the vertebra below it, happening most often at the base of the spine. Spondylolisthesis usually occurs when spondylolysis occurs in the vertebra above, meaning that it has crack or stress fracture.  It commonly occurs in young athletes as well as patients over 60 with osteoarthritis."
    spond_treatment = "Physical therapy exercises to help increase the flexibility in the lower back can help to support abdominal and back muscles, and stretch the hamstrings to alleviate stress in the lower back.\nNonsteroidal anti-inflammatory drugs can help to minimize swelling and pain. In some cases, patients may require a brace to limit spine movement. If severe or high-grade slippage has occurred or symptoms are not alleviating, surgery may be necessary."

    normal_description = ""
    normal_treatment = ""

    if outcome == "Hernia":
        treatment = hernia_treatment
        description = hernia_treatment

    elif outcome == "Normal":
         treatment = normal_treatment
         description = normal_treatment

    elif outcome == "Spondylolisthesis":
         treatment = spond_treatment
         description = spond_treatment

    return jsonify(result=outcome, treatment=treatment, description=description)


@app.route('/')
def index():
    return render_template('minimedic.html')
