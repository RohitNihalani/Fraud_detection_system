#!/bin/bash 
uvicorn main:app --host 0.0.0.0 --port 8001 &

streamlit run app.py --server.port 8002 --server.address 0.0.0.0

wait -n

exit $?
