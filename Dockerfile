FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# # Install dependencies outside of the base image
# RUN DEBIAN_FRONTEND=noninteractive apt update && \
# 	apt install -y --no-install-recommends automake \
#     build-essential  \
#     ca-certificates  \
#     libfreetype6-dev  \
#     libtool  \
#     pkg-config  \
#     python-dev  \
#     python-distutils-extra \
#     cmake \
# 	&& \
#     rm -rf /var/lib/apt/lists/*

RUN pip install matplotlib==3.10.1 \
    mediapipe==0.10.21 \
    numpy==1.26.4 \
    opencv-contrib-python==4.11.0.86 \
    opencv-python==4.11.0.86 \
    opencv-python-headless==4.11.0.86 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    seaborn==0.13.2 \
    tqdm==4.67.1 \
    tensorboard==2.19.0

CMD ["python", "train.py"]

