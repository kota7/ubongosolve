## Ubongo Solver App

A browser-based puzzle solver built with FastAPI, htmx, and a small amount of vanilla JavaScript.
You can define an arbitrary board shape and pieces interactively, then solve the puzzle with one click.

The app runs as a single server (no separate frontend build step) and is designed to be deployed on [Google Cloud Run](https://cloud.google.com/run) at near-zero cost.

### Run locally

```shell
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

Then open http://localhost:8080.

### Run locally on Docker

```shell
docker build -t ubongo-solver .
docker run --rm -p 8080:8080 ubongo-solver
```

Then open http://localhost:8080.

### Deploy on Google Cloud Run

Requires the [gcloud CLI](https://cloud.google.com/sdk/docs/install) with a billing-enabled project.

```shell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

ggcloud run deploy ubongo-solver \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --max-instances=1 \
  --concurrency=10 \
  --timeout=30 \
  --memory=512Mi \
  --cpu=1
```

The free tier of Cloud Run is generous enough that a hobby-level app like this is unlikely to incur any charges.
It is recommended to set up a [budget alert](https://cloud.google.com/billing/docs/how-to/budgets) as a safety net.