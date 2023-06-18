FROM python:3.10

COPY . .

RUN pip3 install -r requirements.txt

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]