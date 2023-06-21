FROM python:3.9-slim-buster

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

# install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

# gunicorn
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "run:app"]
