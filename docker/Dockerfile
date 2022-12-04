FROM continuumio/miniconda3:4.10.3p0-alpine

COPY . /app
WORKDIR /app

RUN conda create --name squad_nlp --file requirements.txt && conda clean --all -y && conda init bash
SHELL ["conda", "run","--no-capture-output", "-n", "squad_nlp", "/bin/bash", "-c"]
EXPOSE 16003

CMD python3 app.py






