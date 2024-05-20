FROM python:3.11 AS builder

RUN pip install pipenv

WORKDIR /srv/app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --ignore-pipfile

COPY src/ src/

COPY index.py .

CMD ["pipenv", "run", "dev"]

# CMD ["tail", "-f", "/dev/null"]