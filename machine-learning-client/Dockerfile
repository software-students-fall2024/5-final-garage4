# use an official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install essential build tools (compiler, linker, and make utility) for compiling dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
 && rm -rf /var/lib/apt/lists/*
 
# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK and download punkt tokenizer data
RUN pip install --no-cache-dir nltk
RUN python -m nltk.downloader punkt punkt_tab

# Install spaCy and download the English model
RUN python -m spacy download en_core_web_sm

# Copy the application files into the container
COPY . .

# Run the main Python script
CMD ["python", "app.py"]