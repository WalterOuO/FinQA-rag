FROM python:3.10-slim

# 設定前端container內工作目錄
WORKDIR /frontend

# 升級 pip, 並直接指定安裝前端僅需的 streamlit 與 requests
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit requests

# 複製前端程式碼
COPY ./frontend /frontend

# 預留 Streamlit 預設端口
EXPOSE 8501
