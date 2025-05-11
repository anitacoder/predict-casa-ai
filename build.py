from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

data = {
    'Height': np.random.uniform(10, 100, 100),
    'prices': np.random.uniform(100000, 1000000, 100)
}

df = pd.DataFrame(data)

x = df[['Height']]
y = df[['prices']]

model = LinearRegression()
model.fit(x,y)

@app.route('/predict', methods=['GET'])
def predict():
    height = float(request.args.get('height'))
    predicted_price =  float(model.predict([[height]])[0][0])
    return jsonify({'price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True) 