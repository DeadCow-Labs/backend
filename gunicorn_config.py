# gunicorn.conf.py
workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'
timeout = 600  # 5 minutes
max_requests = 1000
max_requests_jitter = 50
worker_connections = 1000