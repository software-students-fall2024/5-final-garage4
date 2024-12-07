![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![Machine Learning Client Build](https://github.com/software-students-fall2024/5-final-garage4/actions/workflows/ml-client.yml/badge.svg)
![Web App Build](https://github.com/software-students-fall2024/5-final-garage4/actions/workflows/web-app.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.


## Team Members

- [Yuhao Sheng (ys4689)](https://github.com/imyhalex)
- [Ryoma Nagano (rn2247)](https://github.com/RYOMA-NAGANO)
- [Qiyun Yin (qy765)](https://github.com/Bryccce)
- [Andrea Tang (xt2073)](https://github.com/AndreaTang123)

## Description

The `AI Sentence Checker` is a containerized system designed to provide an intuitive interface for analyzing the sentiment or emotion within a user-provided text. This project combines machine learning, web development, and data visualization.



#### NOTE:
This project uses keyboard and microphone as sensors to detect user input text. There is a sample [`speech.txt`](https://github.com/software-students-fall2024/4-containers-garage3/blob/main/speech1.txt) and [`speech1.txt`](https://github.com/software-students-fall2024/4-containers-garage3/blob/main/speech.txt) and you can copy and paste them into the input text box to have a quick look. Otherwise you need type in or find a source text that is more than 300 words to ensure you get the analyze result from ml client.

## How to run

___1. Make sure Docker Desktop is installed in your local machine___
> - [link to download the Docker Desktop](https://www.docker.com/products/docker-desktop/)
Note: Make sure you also installed [Docker Compose](https://docs.docker.com/compose/)

___2. Clone this repository to your local machine___
```text
https://github.com/software-students-fall2024/4-containers-garage3.git
```

___3. Create virtual environments for both web app and machine learning client___
```bash
# for web-app
$ cd web-app
$ python3 -m venv .venv

# for ml client
$ cd machine-learning-client
$ python3 -m venv .venv
```
___4. Activate a virtual environment and install required packages (machine-learning-client in this example)___
```bash
project@root$ cd machine-learning-client
project@root/machine-learning-client$ source .venv/bin/activate
project@root/machine-learning-client$ pip3 install -r requirements.txt

# note: if you want to run web-app locally, you need to deactivate the ml's virtual environment first and do the aforementioned step again
```
___5. Before containerized to your docker, you can test each service part locally to see how it interact with the database by adding your MONGO_URI to `.env`(machine-learning-client in this example)___
```bash
# let's say current dir is in: 4-containers-garage3/machine-learning-client

# create a .env file
touch .env

# within the .env enter:
MONOGO_URI=mongodb://mongo:27017/ # default uri
```

___6. Build your Docker images___
```bash
$ docker-compose up --build

# run the service in the background in detached mode:
$ docker-compose up -d

# shut down and remove the container
$ docker-compose down
```
___7. Go to Docker Desktop, click on the 5000:5000 port to run the webpage___

## Docker Hub Images
[machine-learning-client](https://hub.docker.com/r/imyhalex/machine-learning-client)

[web-app](https://hub.docker.com/r/imyhalex/web-app)

## Digital Ocean

website url:
```text
http://104.131.13.180:5000/
```