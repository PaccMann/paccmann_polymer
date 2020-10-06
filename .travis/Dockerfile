FROM drugilsberg/rdkit-ubuntu:latest
RUN apt-get update && apt-get install -y git
WORKDIR /paccmann_polymer
# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY pytorch_geometric_install.sh .
RUN /bin/bash ./pytorch_geometric_install.sh cpu
# copy paccmann_polymer
COPY . .
RUN pip3 install --no-deps .
CMD /bin/bash
