FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-dev git vim

RUN pip install ipdb jupyter ipython opencv-python
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV CUDA_HOME="/usr/local/cuda"
ENV FORCE_CUDA="1"
RUN git clone https://github.com/aim-uofa/AdelaiDet.git
RUN cd AdelaiDet && python setup.py build develop

#RUN rm -rf AdelaiDet
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /tmp/*
RUN rm -rf ~/.cache
RUN apt clean all
RUN conda clean -y -a

EXPOSE 8888
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8