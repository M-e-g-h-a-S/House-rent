from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and columns
model = joblib.load('house_rent_model.pkl')
model_columns = joblib.load('model_columns.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_input(df):
    # Scale 'Size'
    df['Size'] = scaler.transform(df[['Size']])

    # Manually encode city columns including 'Bangalore'
    city_map = {
        "Bangalore": [1, 0, 0, 0, 0, 0],
        "Chennai":   [0, 1, 0, 0, 0, 0],
        "Delhi":     [0, 0, 1, 0, 0, 0],
        "Hyderabad": [0, 0, 0, 1, 0, 0],
        "Kolkata":   [0, 0, 0, 0, 1, 0],
        "Mumbai":    [0, 0, 0, 0, 0, 1]
    }

    city_cols = ['City_1', 'City_2', 'City_3', 'City_4', 'City_5', 'City_6']
    city_values = city_map.get(df.at[0, 'City'], [0, 0, 0, 0, 0, 0])

    for col, val in zip(city_cols, city_values):
        df[col] = val

    df.drop('City', axis=1, inplace=True)

    # Ensure correct column order
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

@app.route('/')
def home():
    return '''
        <h2>House Rent Predictor</h2>
        <form action="/predict" method="post">
            Size (in sq ft): <input type="number" name="Size"><br>
            City: 
            <select name="City">
                <option value="Bangalore">Bangalore</option>
                <option value="Chennai">Chennai</option>
                <option value="Delhi">Delhi</option>
                <option value="Hyderabad">Hyderabad</option>
                <option value="Kolkata">Kolkata</option>
                <option value="Mumbai">Mumbai</option>
            </select><br><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = {
                'Size': float(request.form['Size']),
                'City': request.form['City']
            }
        df = pd.DataFrame([data])
        processed_df = preprocess_input(df)
        prediction = model.predict(processed_df)
        return jsonify({'predicted_rent': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)