from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and supporting files
model = joblib.load('house_rent_model.pkl')
model_columns = joblib.load('model_columns.pkl')
scaler = joblib.load('scaler.pkl')

# Mapping
city_map = {
    "Bangalore": [0, 0, 0, 0, 0],  # will be dropped
    "Chennai":   [1, 0, 0, 0, 0],
    "Delhi":     [0, 1, 0, 0, 0],
    "Hyderabad": [0, 0, 1, 0, 0],
    "Kolkata":   [0, 0, 0, 1, 0],
    "Mumbai":    [0, 0, 0, 0, 1]
}

furnishing_map = {
    "Unfurnished": 0,
    "Semi-Furnished": 1,
    "Furnished": 2
}

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_input(df):
    # Scale Size
    df['Size'] = scaler.transform(df[['Size']])

    # Encode City
    city_vals = city_map.get(df.at[0, 'City'], [0, 0, 0, 0, 0])
    for i in range(1, 6):
        df[f'City_{i}'] = city_vals[i - 1]
    df.drop('City', axis=1, inplace=True)

    # Reorder columns
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
    'BHK': int(request.form['bhk']),
    'Size': float(request.form['size']),
    'Furnishing Status': furnishing_map[request.form['furnishing']],
    'City': request.form['city']
}

        df = pd.DataFrame([data])
        processed = preprocess_input(df)
        prediction = model.predict(processed)
        return render_template('index.html', prediction_text=f'Predicted Rent: ₹{round(prediction[0], 2)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
