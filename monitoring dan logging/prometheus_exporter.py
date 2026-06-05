import mlflow.pyfunc
from prometheus_client import start_http_server, Counter, Histogram
import time


class MonitoredWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, autolog_model_uri):
        self.autolog_model_uri = autolog_model_uri

    def load_context(self, context):
        # 1. Load your original autologged model artifact
        self.model = mlflow.pyfunc.load_model(self.autolog_model_uri)

        # 2. Spin up the Prometheus scrape endpoint on local port 8000
        start_http_server(8000)
        print("--> Prometheus exporter listening locally on http://localhost:8000")

        # 3. Define your custom monitoring metrics
        self.prediction_counter = Counter('model_predictions_total', 'Total inferences served')
        self.latency_histogram = Histogram('model_prediction_latency_seconds', 'Inference latency')

    def predict(self, context, model_input):
        start_time = time.time()
        self.prediction_counter.inc()

        # Pass the input directly to your autologged model's prediction method
        predictions = self.model.predict(model_input)

        duration = time.time() - start_time
        self.latency_histogram.observe(duration)

        return predictions


if __name__ == "__main__":
    print("memulai pembacaan model")
    # arahkan variabel ini ke path absolut dari ID run yang ingin digunakan
    AUTOLOG_MODEL_URI = "D:/david/coding programs/dicoding/Pijak IBM SkillsBuild 2025-2026/tugas tech skill/membangun sistem machine learning/membangun_model/mlruns/920067890604096236/5f39165986474917b374be611f717914/artifacts/model"

    # Instantiate our wrapper
    wrapped_model = MonitoredWrapper(autolog_model_uri=AUTOLOG_MODEL_URI)

    # Save the wrapped model into a local folder
    output_dir = "monitored_model_local"
    mlflow.pyfunc.save_model(path=output_dir, python_model=wrapped_model)
    print(f"Success! Monitored wrapper saved to folder: ./{output_dir}")
