FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

RUN mkdir -p /app
WORKDIR /app

COPY requirements_image.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.6/autoawq-0.1.6+cu118-cp310-cp310-linux_x86_64.whl

COPY app.py app.py
COPY gpu_test.py gpu_test.py

#RUN mkdir -p ~/.cache/huggingface/hub
#COPY models--01-ai--Yi-6B-Chat-4bits /root/.cache/huggingface/hub/models--01-ai--Yi-6B-Chat-4bits

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8000"]

# docker image build -t richardxgf/yi-llm:1.0.0 .
# docker run --rm --gpus all -p 8000:8000 -it  richardxgf/yi-llm:1.0.0
# docker rmi $(docker images -f dangling=true -q)