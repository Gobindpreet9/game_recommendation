# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and common data science packages
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    chromadb \
    sentence-transformers \
    torch \
    transformers \
    langchain \
    kagglehub

# Make port 8888 available for Jupyter
EXPOSE 8888

# Create a non-root user to run Jupyter
RUN useradd -m jupyter
USER jupyter

# Set up working directory for the user
WORKDIR /home/jupyter

# Start Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]