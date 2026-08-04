# 星野爱拟人 Agent - Docker 镜像
FROM python:3.10-slim

WORKDIR /app

# 先复制依赖文件并安装，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露服务端口
EXPOSE 8001

# 启动 FastAPI 服务（非 reload 模式，对外可访问）
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
