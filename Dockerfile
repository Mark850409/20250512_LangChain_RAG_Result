# 使用輕量級 Python 映像檔
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 複製並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製主要應用程式
COPY . .

# 暴露端口（配合環境變數）
EXPOSE 5000

# 設定環境變數
ENV PORT=5000

# 設定啟動 FastAPI 應用
CMD ["python", "app.py"]
