# src/api.py

import logging
import time
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
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
CACHE_HITS = Counter('recommendations_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('recommendations_cache_misses_total', 'Cache misses')
CACHE_SIZE = Gauge('recommendations_cache_size', 'Cache size')


class UserRequest(BaseModel):
    user_id: int
    k: int = 10


class RecommendationService:
    def __init__(self):
        self.ranker = None
        self.als_recommendations = None
        self.top_popular = None
        self.served_users = set()
        # Кэш для рекомендаций
        self._cache = {}
        self._cache_ttl = {}  # Time to live
        self.cache_duration = 3600  # 1 час

    def load(self):
        start_time = time.time()
        logger.info("Loading models...")
        
        self.als_recommendations = pd.read_parquet(
            f"s3://{bucket_name}/recsys/recommendations/final_recommendations.parquet",
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

    def _get_cache_key(self, user_id: int, k: int) -> str:
        """Генерация ключа для кэша"""
        return f"user_{user_id}_k_{k}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверка валидности кэша"""
        if cache_key not in self._cache_ttl:
            return False
        return time.time() - self._cache_ttl[cache_key] < self.cache_duration

    def get_recommendations(self, user_id: int, k: int = 10):
        # Проверяем кэш
        cache_key = self._get_cache_key(user_id, k)
        
        if cache_key in self._cache and self._is_cache_valid(cache_key):
            REQUEST_COUNT.labels(type="cached").inc()
            CACHE_HITS.inc()
            logger.debug(f"Cache HIT for user_id={user_id}")
            return self._cache[cache_key]
        
        CACHE_MISSES.inc()
        
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
            recs = user_candidates.nlargest(k, 'final_score')['item_id_enc'].tolist()
        
        # Сохраняем в кэш
        self._cache[cache_key] = recs
        self._cache_ttl[cache_key] = time.time()
        CACHE_SIZE.set(len(self._cache))
        logger.debug(f"Cache MISS for user_id={user_id}, cached result")
        
        RECOMMENDATIONS_RETURNED.observe(len(recs))
        return recs

    def clear_expired_cache(self):
        """Очистка устаревшего кэша"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_ttl.items()
            if current_time - timestamp > self.cache_duration
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        
        if expired_keys:
            CACHE_SIZE.set(len(self._cache))
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")


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


@app.post("/cache/clear")
async def clear_cache():
    """Очистка устаревшего кэша (для админов)"""
    rec_service.clear_expired_cache()
    return {
        "status": "ok", 
        "cache_size": len(rec_service._cache),
        "message": "Expired cache entries cleared"
    }


@app.get("/cache/stats")
async def cache_stats():
    """Статистика кэша"""
    return {
        "cache_size": len(rec_service._cache),
        "cache_duration_seconds": rec_service.cache_duration,
        "unique_users_served": len(rec_service.served_users)
    }