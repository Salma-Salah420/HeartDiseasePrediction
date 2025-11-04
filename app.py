from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc

# 🔹 حمّل الموديل من مجلد MLflow artifacts
MODEL_PATH = "mlruns/194489145900410023/models/m-9cd419fc238646248d7d87bf154a7713/artifacts"
model = mlflow.pyfunc.load_model(MODEL_PATH)

# 🔹 أنشئ تطبيق Flask
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "✅ Heart Disease MLflow model is running!"})

@app.route("/invocations", methods=["POST"])
def predict():
    try:
        # استقبل البيانات بنفس تنسيق MLflow serve
        data = request.get_json()

        if "dataframe_split" not in data:
            return jsonify({"error": "Invalid JSON format. Expected 'dataframe_split' key."}), 400

        df = pd.DataFrame(data["dataframe_split"]["data"],
                          columns=data["dataframe_split"]["columns"])

        # 🔹 تنبؤ بالموديل
        predictions = model.predict(df)

        # 🔹 أعد النتائج
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
