FROM continuumio/miniconda3
RUN apt-get update && \
    apt-get install -y build-essential git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app

RUN conda create -n custom-env -c rdkit -c conda-forge python=3.7  rdkit umap-learn
RUN echo "source activate custom-env" > ~/.bashrc
ENV PATH /opt/conda/envs/custom-env/bin:$PATH

COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt
COPY . /usr/src/app
# other installation steps
EXPOSE 80
# or whatever is the name of the dash app
CMD ["python3", "app.py"]
