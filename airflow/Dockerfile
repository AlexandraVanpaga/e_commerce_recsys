FROM apache/airflow:2.7.3-python3.10

USER root


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow


COPY requirements.txt ./tmp/requirements.txt
RUN pip install -U pip
RUN pip install -r ./tmp/requirements.txt
