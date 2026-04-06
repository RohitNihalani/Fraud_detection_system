FROM python:3.11-slim

WORKDIR /app

COPY  . .

RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh


EXPOSE 8001
EXPOSE 8002

CMD [ "./entrypoint.sh"]
