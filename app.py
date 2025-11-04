from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import os

# 🔹 تحديد المسار النسبي للنموذج داخل الحاوية
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

# ✅ تحميل الموديل مرة واحدة عند بدء التشغيل
model = mlflow.pyfunc.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "✅ Heart Disease MLflow model is running!"})

@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "dataframe_split" not in data:
            return jsonify({"error": "Invalid JSON format. Expected 'dataframe_split' key."}), 400

        df = pd.DataFrame(
            data["dataframe_split"]["data"],
            columns=data["dataframe_split"]["columns"]
        )

        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
