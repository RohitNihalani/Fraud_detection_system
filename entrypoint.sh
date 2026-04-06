#!/bin/bash 
python main.py &

python app.py

wait -n

exit $?
