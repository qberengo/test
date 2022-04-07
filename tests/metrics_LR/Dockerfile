FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y && pip3 install fastapi && pip3 install requests
ADD api_metrics_LR.py /data/api_metrics_LR.py
VOLUME /data/
WORKDIR /data/
CMD python3 api_metrics_LR.py
