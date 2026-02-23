后端启动
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
前端启动npm run dev