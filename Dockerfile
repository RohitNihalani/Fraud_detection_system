FROM python:3.11-slim

WORKDIR /app

COPY  . .

RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh


EXPOSE 8000
EXPOSE 8001

CMD [ "./entrypoint.sh"]
