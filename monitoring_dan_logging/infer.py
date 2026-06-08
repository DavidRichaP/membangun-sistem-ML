import pandas as pd
import requests

# endpoint listen model
URL = "http://localhost:5001/invocations"

df = pd.read_csv("gender_classif_v7_preprocessed.csv")

if "quality" in df.columns:
    df = df.drop(columns=["gender"])

features = [
    "forehead_width_cm",
    "forehead_height_cm",
    "long_hair",
    "nose_wide",
    "nose_long",
    "lips_thin",
    "distance_nose_to_lip_long"
]

df = df[features]

# format ke mlflow split orientation dictionary
payload = {
    "dataframe_split": df.to_dict(orient="split")
}

# mengirim request ke http server
try:
    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, json=payload, headers=headers)

    if response.status_code == 200:
        print("🚀 Inference Successful!")
        predictions = response.json()
        print("Predicted Quality Outputs:", predictions)
    else:
        print(f"❌ Server returned error code {response.status_code}")
        print("Details:", response.text)

except requests.exceptions.ConnectionError:
    print("❌ Connection Failed! Is your MLflow server running on port 5000?")
