from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# تحميل الموديل
model_path = os.path.join("model", "model.pkl")
model = joblib.load(model_path)

# قائمة الأعمدة المطلوبة (يجب أن تتطابق تمامًا مع التدريب)
FEATURE_COLUMNS = [
    'Unnamed: 0', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
    'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
    'SkinCancer', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other',
    'Race_White', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
    'Diabetic_Yes (during pregnancy)'
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ MLflow Heart Disease Prediction API is running successfully!",
        "usage": "Send POST request to /predict with full JSON feature names."
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # التأكد من وجود القيم المطلوبة
        missing = [f for f in FEATURE_COLUMNS if f not in data and f != 'Unnamed: 0']
        if missing:
            return jsonify({
                "error": f"Missing features: {missing}"
            }), 400

        # إنشاء DataFrame بصف واحد من القيم
        df = pd.DataFrame([data], columns=[f for f in FEATURE_COLUMNS if f != 'Unnamed: 0'])

        # ملء أي قيم NaN بـ 0
        df = df.fillna(0)

        # التنبؤ
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(proba)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
