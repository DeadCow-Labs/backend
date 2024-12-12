# gunicorn.conf.py
workers = 2
worker_class = 'uvicorn.workers.UvicornWorker'
timeout = 300  # 5 minutes
max_requests = 1000
max_requests_jitter = 50