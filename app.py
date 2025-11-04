from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ تحميل الموديل من المجلد الصحيح
model = joblib.load("model/model.pkl")

# ✅ ترتيب الأعمدة كما تم تدريب النموذج عليها
expected_features = [
    'Unnamed: 0', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
    'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
    'SkinCancer', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other',
    'Race_White', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
    'Diabetic_Yes (during pregnancy)'
]

@app.route('/')
def home():
    return jsonify({
        "message": "✅ MLflow Heart Disease Prediction API is running successfully!",
        "usage": "Send POST request to /predict with JSON data."
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # تحويل البيانات إلى DataFrame بالترتيب الصحيح
        df = pd.DataFrame([data], columns=expected_features)

        prediction = model.predict(df)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
