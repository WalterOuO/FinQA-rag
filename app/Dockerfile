FROM python:3.10-slim

# 設定貨櫃內工作目錄
WORKDIR /app

# 安裝 PaddleOCR、pdfplumber 與 OpenCV 所需的 Linux 系統底層依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements.txt 利用 Docker 快取層
# 💡 注意：因為 requirements.txt 在 app 的上一層，
# 之後 build 時建議將 Context 設在 FinQA-rag-project 專案根目錄
COPY requirements.txt .

# 升級 pip 並安裝所有後端與 RAG 核心套件
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製 app 資料夾內的所有原始碼到貨櫃中
COPY ./app /app

# 設定 Python 模組搜尋路徑，確保跨資料夾引入 (__init__.py) 正常運作
ENV PYTHONPATH=/app

# 預留端口 (FastAPI 使用，Celery Worker 會自動忽略)
EXPOSE 8000
