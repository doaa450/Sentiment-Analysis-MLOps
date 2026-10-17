# Sentiment-Analysis

## Completion checklists

### P R O J E C T 1 · D E E P L E A R N I N G — C O D E , P A C K A G I N G , T R A C K I N G , V E R S I O N I N G

✅ 1- src/ layout with pyproject.toml — pip install -e . works

☐ 2- AraBERT or CAMeL-BERT loaded in a class with type hints

☐ 3- FastAPI /predict : {text: str} → {label, confidence, model_version}

☐ 4- Pydantic rejects empty text — 422 returned, tested with curl

☐ 5- /health returns {status: healthy} — used by the Docker healthcheck

☐ 6- docker build -t arabic-sentiment . passes; docker compose up starts the service on port 8000

☐ 7- README: exactly 3 commands to run on any machine

☐ 8- MLflow tracking server running — UI accessible

☐ 9- ≥ 5 runs: model_name , lr , batch_size , accuracy , f1_macro

☐ 10- MLflow comparison screenshot in /reports/mlflow_comparison.png

☐ 11- Best model registered as ArabicSentiment → promoted to Production

☐ 12- Labeled dataset tracked with DVC — dvc status shows clean

☐ 13- dvc repro reproduces training with the same metrics (± 0.005)

☐ 14- GitHub Actions: lint → test → docker build → push on main

☐ 15- CI fails if f1_macro drops below baseline — documented in README


### P R O J E C T 2 · L L M / R A G — C O D E , P A C K A G I N G , T R A C K I N G , V E R S I O N I N G

☐ 1- BentoML service with batchable=True Runner — bentoml models list shows the model
☐ 2- bentoml serve runs /predict at localhost:8000
☐ 3- Locust report before optimization in /reports/locust_fastapi.html
☐ 4- TensorRT FP16 engine built — committed or documented
☐ 5- Locust after TensorRT in /reports/locust_trt.html — p95 in README
☐ 6- Batch scoring: ≥ 1,000 samples, output to /data/scoring/output/
☐ 7- nginx canary config: weight 5/95 — rollout stages in README
☐ 8- Evidently DataDrift report on the latest batch — saved in /reports/
☐ 9- PSI computed for text_length and confidence_score features
☐ 10- Drift check triggered after every batch scoring run
☐ 11- Prometheus exposes the PSI score at /metrics
☐ 12- Grafana: ≥ 2 panels (p95 + PSI) — screenshot in README
☐ 13- Alert: PSI > 0.25 → notification documented in README
☐ 14- 12-layer AraBERT distilled to a 6-layer student — model file committed
☐ 15- INT8 ONNX quantization applied — model_int8.onnx committed
☐ 16- Benchmark table: original vs distilled vs INT8 (accuracy, p95, size)
☐ 17- Final architecture diagram in README covering all 5 sessions
