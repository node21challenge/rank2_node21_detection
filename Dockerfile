FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update
RUN apt-get -y install gcc

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN  pip install --upgrade pip

# Copy all required files so that they are available within the docker image 
# All the codes, weights, anything you need to run the algorithm!
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/

COPY --chown=algorithm:algorithm utils /opt/algorithm/utils
COPY --chown=algorithm:algorithm models /opt/algorithm/models
COPY --chown=algorithm:algorithm data /opt/algorithm/data

COPY --chown=algorithm:algorithm yolo1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo3.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo4.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo5.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo6.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo7.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo8.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo9.pt /opt/algorithm/

COPY --chown=algorithm:algorithm yolo1-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo2-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo3-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo4-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo5-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo6-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo7-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo8-1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo9-1.pt /opt/algorithm/

COPY --chown=algorithm:algorithm yolo1-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo2-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo3-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo4-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo5-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo6-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo7-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo8-2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo9-2.pt /opt/algorithm/

COPY --chown=algorithm:algorithm yolo1f1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo1f2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo2f1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo2f2.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo3f1.pt /opt/algorithm/
COPY --chown=algorithm:algorithm yolo3f2.pt /opt/algorithm/

COPY --chown=algorithm:algorithm nodule01.yaml /opt/algorithm/

# Install required python packages via pip - please see the requirements.txt and adapt it to your needs
RUN python -m pip install --user -rrequirements.txt
RUN ls /opt/algorithm/

COPY --chown=algorithm:algorithm omegaconf-2.1.1-py3-none-any.whl /opt/algorithm
RUN python -m pip install omegaconf-2.1.1-py3-none-any.whl

COPY --chown=algorithm:algorithm iopath-0.1.9-py3-none-any.whl /opt/algorithm
RUN python -m pip install iopath-0.1.9-py3-none-any.whl

COPY --chown=algorithm:algorithm fvcore-0.1.5.post20211023.tar.gz /opt/algorithm
RUN python -m pip install fvcore-0.1.5.post20211023.tar.gz

COPY --chown=algorithm:algorithm pycocotools-2.0.3.tar.gz /opt/algorithm
RUN python -m pip install pycocotools-2.0.3.tar.gz

COPY --chown=algorithm:algorithm detectron2-0.6+cu111-cp37-cp37m-linux_x86_64.whl /opt/algorithm/ 
RUN python -m pip install detectron2-0.6+cu111-cp37-cp37m-linux_x86_64.whl


COPY --chown=algorithm:algorithm process.py /opt/algorithm/


# Entrypoint to run, entypoint.sh files executes process.py as a script
ENTRYPOINT ["bash", "entrypoint.sh"]

## ALGORITHM LABELS: these labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=noduledetection
# These labels are required and describe what kind of hardware your algorithm requires to run for grand-challenge.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=12G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=10G
