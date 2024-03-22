# Use an official Ubuntu base image
FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Python and Pip
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


# Install local codebase
RUN pip3 install -e .

EXPOSE 5000

# TODO Ollama install within container
# TODO preload model weights for Llama2

## Define environment variable
#ENV NAME World

# Run app.py when the container launches
CMD ["python3", "main.py"]
