FROM nekodigi/gpu_essentials

WORKDIR /app

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin


COPY requirements.txt requirements.txt

#install pip
#RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install -r requirements.txt

#RUN python -c "from opencv_fixer import AutoFix; AutoFix()"



WORKDIR /root
COPY utils.sh utils.sh
RUN echo "source ~/utils.sh" >> ~/.zshrc

