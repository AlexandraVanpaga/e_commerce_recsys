import logging
import time
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRanker
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os
from dotenv import load_dotenv

logger = logging.getLogger("uvicorn.error")

load_dotenv()

endpoint_url = os.getenv("S3_ENDPOINT_URL")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("S3_BUCKET_NAME")

storage_options = {
    'key': aws_access_key_id,
    'secret': aws_secret_access_key,
    'client_kwargs': {'endpoint_url': endpoint_url}
}

# === PROMETHEUS МЕТРИКИ ===
REQUEST_COUNT = Counter('recommendations_requests_total', 'Total requests', ['type'])
REQUEST_LATENCY = Histogram('recommendations_latency_seconds', 'Request latency', 
                            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
RECOMMENDATIONS_RETURNED = Histogram('recommendations_returned_count', 'Recommendations count',
                                     buckets=[1, 5, 10, 20, 50, 100])
ERRORS_COUNT = Counter('recommendations_errors_total', 'Total errors', ['error_type'])
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Model load time')
DATA_SIZE = Gauge('data_size_rows', 'Data size', ['data_type'])
USERS_SERVED = Counter('users_served_total', 'Total unique users served')


class UserRequest(BaseModel):
    user_id: int
    k: int = 10


class RecommendationService:
    def __init__(self):
        self.ranker = None
        self.als_recommendations = None
        self.top_popular = None
        self.served_users = set()

    def load(self):
        start_time = time.time()
        logger.info("Loading models...")
        
        self.ranker = CatBoostRanker()
        self.ranker.load_model("/models/prod_model_catboost_ranker.cbm")
        logger.info("CatBoost loaded")
        
        self.als_recommendations = pd.read_parquet(
            f"s3://{bucket_name}/recsys/recommendations/personal_als.parquet",
            storage_options=storage_options
        )
        DATA_SIZE.labels(data_type="als_recommendations").set(len(self.als_recommendations))
        logger.info(f"ALS loaded: {len(self.als_recommendations):,}")
        
        self.top_popular = pd.read_parquet(
            f"s3://{bucket_name}/recsys/recommendations/top_popular.parquet",
            storage_options=storage_options
        )
        DATA_SIZE.labels(data_type="top_popular").set(len(self.top_popular))
        logger.info(f"Top popular loaded: {len(self.top_popular)}")
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        logger.info(f"Models loaded in {load_time:.2f}s")

    def get_recommendations(self, user_id: int, k: int = 10):
        # Трекаем уникальных пользователей
        if user_id not in self.served_users:
            self.served_users.add(user_id)
            USERS_SERVED.inc()
        
        user_candidates = self.als_recommendations[
            self.als_recommendations['user_id_enc'] == user_id
        ].copy()
        
        if len(user_candidates) == 0:
            REQUEST_COUNT.labels(type="default").inc()
            recs = self.top_popular['item_id_enc'].head(k).tolist()
        else:
            REQUEST_COUNT.labels(type="personal").inc()
            recs = user_candidates.nlargest(k, 'score')['item_id_enc'].tolist()
        
        RECOMMENDATIONS_RETURNED.observe(len(recs))
        return recs


rec_service = RecommendationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting...")
    rec_service.load()
    yield
    logger.info("Stopping...")


app = FastAPI(title="E-commerce Recommendations", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommendations")
async def recommendations(req: UserRequest):
    try:
        with REQUEST_LATENCY.time():
            recs = rec_service.get_recommendations(req.user_id, req.k)
        return {"user_id": req.user_id, "recommendations": recs}
    except Exception as e:
        logger.error(f"Error user_id={req.user_id}: {e}")
        ERRORS_COUNT.labels(error_type="recommendation_error").inc()
        return {"user_id": req.user_id, "recommendations": rec_service.top_popular['item_id_enc'].head(req.k).tolist()}