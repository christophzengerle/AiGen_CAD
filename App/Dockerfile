FROM python:3.10

WORKDIR /usr/app/src/App
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# CMD ["python", "app.py"]
