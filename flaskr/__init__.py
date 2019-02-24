import os

from flask import Flask, render_template

app = Flask(__name__)
app.run(debug=True)


@app.route('/hello')
def hello():
    return render_template('data.html')
