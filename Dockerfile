FROM nekodigi/gpu_essentials

WORKDIR /app


COPY requirements.txt requirements.txt

#install pip
#RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install -r requirements.txt

#RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
