from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Precio de Casa</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 500px; width: 100%; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 25px; font-size: 22px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #555; font-size: 14px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus { border-color: #3498db; outline: none; }
        button { width: 100%; padding: 12px; background: #3498db; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #2980b9; }
        .btn-clear { background: #e74c3c; margin-top: 8px; }
        .btn-clear:hover { background: #c0392b; }
        .result { text-align: center; margin-top: 20px; padding: 20px; background: #d4edda; border-radius: 8px; }
        .result h2 { color: #155724; font-size: 28px; }
        .row { display: flex; gap: 15px; }
        .row .form-group { flex: 1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predicción de Precio de Casa</h1>
        <form method="POST">
            <div class="row">
                <div class="form-group">
                    <label>Pies Cuadrados</label>
                    <input type="number" name="square_feet" step="any" value="{{ vals.square_feet }}" required>
                </div>
                <div class="form-group">
                    <label>Núm. Habitaciones</label>
                    <input type="number" name="num_bedrooms" value="{{ vals.num_bedrooms }}" required>
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Núm. Baños</label>
                    <input type="number" name="num_bathrooms" value="{{ vals.num_bathrooms }}" required>
                </div>
                <div class="form-group">
                    <label>Núm. Pisos</label>
                    <input type="number" name="num_floors" value="{{ vals.num_floors }}" required>
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Año de Construcción</label>
                    <input type="number" name="year_built" value="{{ vals.year_built }}" required>
                </div>
                <div class="form-group">
                    <label>Tamaño de Garaje</label>
                    <input type="number" name="garage_size" step="any" value="{{ vals.garage_size }}" required>
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Puntuación Ubicación</label>
                    <input type="number" name="location_score" step="any" value="{{ vals.location_score }}" required>
                </div>
                <div class="form-group">
                    <label>Distancia al Centro</label>
                    <input type="number" name="distance_to_center" step="any" value="{{ vals.distance_to_center }}" required>
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Tiene Jardín</label>
                    <select name="has_garden">
                        <option value="0" {{ 'selected' if vals.has_garden == '0' }}>No</option>
                        <option value="1" {{ 'selected' if vals.has_garden == '1' }}>Sí</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Tiene Alberca</label>
                    <select name="has_pool">
                        <option value="0" {{ 'selected' if vals.has_pool == '0' }}>No</option>
                        <option value="1" {{ 'selected' if vals.has_pool == '1' }}>Sí</option>
                    </select>
                </div>
            </div>
            <button type="submit">Predecir Precio</button>
            <button type="reset" class="btn-clear">Limpiar</button>
        </form>
        {% if prediction %}
        <div class="result">
            <p>Precio estimado:</p>
            <h2>${{ prediction }}</h2>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    vals = {
        'square_feet': '', 'num_bedrooms': '', 'num_bathrooms': '', 'num_floors': '',
        'year_built': '', 'garage_size': '', 'location_score': '', 'distance_to_center': '',
        'has_garden': '0', 'has_pool': '0'
    }
    if request.method == 'POST':
        vals = request.form.to_dict()
        features = np.array([[
            float(vals['square_feet']),
            int(vals['num_bedrooms']),
            int(vals['num_bathrooms']),
            int(vals['num_floors']),
            int(vals['year_built']),
            int(vals['has_garden']),
            int(vals['has_pool']),
            float(vals['garage_size']),
            float(vals['location_score']),
            float(vals['distance_to_center'])
        ]])
        features_scaled = scaler.transform(features)
        result = model.predict(features_scaled)[0]
        prediction = f"{result:,.2f}"
    return render_template_string(HTML, prediction=prediction, vals=vals)

if __name__ == '__main__':
    app.run(debug=True)