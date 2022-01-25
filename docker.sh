docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size=2gb -v $(pwd):/workspace --rm -it docker.io/deepspeed/deepspeed:latest-torch170-cuda110
# docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size=2gb -v $(pwd):/workspace --rm -it nvcr.io/nvidia/pytorch:20.06-py3

