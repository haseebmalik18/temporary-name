import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='.')
CORS(app)

respiratory_forecast = joblib.load(os.path.join(BASE_DIR, 'models/respiratory_disease_rate_forecaster.pkl'))
heat_forecast = joblib.load(os.path.join(BASE_DIR, 'models/heat_related_admissions_forecaster.pkl'))
country_encoder = joblib.load(os.path.join(BASE_DIR, 'models/country_encoder.pkl'))

COUNTRY_CLASSES = list(country_encoder.classes_)

df = pd.read_csv(os.path.join(BASE_DIR, 'cleaned_file.csv'))
country_meta = {}
for _, r in df.iterrows():
    c = r['country_name']
    if c not in country_meta:
        country_meta[c] = {
            'lat': r['latitude'], 'lon': r['longitude'],
            'access': r['healthcare_access_index'], 'temps': {}
        }
    m = int(r['month'])
    if m not in country_meta[c]['temps']:
        country_meta[c]['temps'][m] = []
    country_meta[c]['temps'][m].append(r['temperature_celsius'])

forecast_cache = {}
for c in COUNTRY_CLASSES:
    meta = country_meta.get(c)
    if not meta:
        continue
    country_enc = COUNTRY_CLASSES.index(c)
    results = []
    for m in range(1, 13):
        temps = meta['temps'].get(m, [15])
        temp = sum(temps) / len(temps)
        month_sin = np.sin(2 * np.pi * m / 12)
        month_cos = np.cos(2 * np.pi * m / 12)
        features = np.array([[country_enc, month_sin, month_cos, meta['lat'], meta['lon'], temp, meta['access']]])
        results.append({
            'month': m,
            'heat_related_admissions': round(max(0, float(heat_forecast.predict(features)[0])), 2),
            'respiratory_disease_rate': round(max(0, float(respiratory_forecast.predict(features)[0])), 2)
        })
    forecast_cache[c] = results


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/forecast/all', methods=['GET'])
def forecast_all():
    return jsonify(forecast_cache)


@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    app.run(port=5001, debug=False)
