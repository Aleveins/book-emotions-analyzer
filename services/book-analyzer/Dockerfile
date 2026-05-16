FROM python:3.11-slim AS protogen

RUN pip install --no-cache-dir grpcio-tools==1.66.2

WORKDIR /protogen
COPY proto/text_storage/text_storage.proto proto/text_storage/text_storage.proto

RUN mkdir -p /protogen/out \
    && python -m grpc_tools.protoc \
        --proto_path=proto/text_storage \
        --python_out=/protogen/out \
        --grpc_python_out=/protogen/out \
        proto/text_storage/text_storage.proto \
    && ls /protogen/out

# Сгенерированный gRPC-код импортирует `text_storage_pb2` абсолютно; внутри пакета нужен относительный импорт
RUN sed -i 's/^import text_storage_pb2/from . import text_storage_pb2/' /protogen/out/text_storage_pb2_grpc.py


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/book-analyzer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY book_emotion_model_package /app/book_emotion_model_package
ENV PYTHONPATH=/app/book_emotion_model_package:/app

COPY services/book-analyzer/src /app/src

COPY --from=protogen /protogen/out/text_storage_pb2.py /app/src/proto/text_storage_pb2.py
COPY --from=protogen /protogen/out/text_storage_pb2_grpc.py /app/src/proto/text_storage_pb2_grpc.py

CMD ["python", "-m", "src.main"]
