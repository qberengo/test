FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y && pip3 install Flask && pip3 install requests && pip3 install scikit-learn && pip3 install pandas
EXPOSE 5000
ADD app.py /data/app.py
ADD bike.csv /data/bike.csv
ADD credentials.csv /data/credentials.csv
WORKDIR /data/
CMD python3 app.py
