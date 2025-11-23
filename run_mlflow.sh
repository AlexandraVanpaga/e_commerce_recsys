export MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL}
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}


mlflow server \
  --backend-store-uri postgresql://mle_20250729_0060996a6e_freetrack:3c05f7b15a854e81907215f46d411f6d@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250729_0060996a6e \
  --default-artifact-root s3://s3-student-mle-20250729-0060996a6e-freetrack \
  --registry-store-uri postgresql://mle_20250729_0060996a6e_freetrack:3c05f7b15a854e81907215f46d411f6d@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250729_0060996a6e \
  --host 0.0.0.0 \
  --port 5000 \
  --no-serve-artifacts