# Use official Python image as base image
FROM python:3.10.5-slim

# Set environment variables to avoid interactive prompts during package installations
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION=1.9.0

# Set working directory inside the container
WORKDIR /app

# Copy the poetry configuration files and project files
COPY pyproject.toml /app/

RUN pip install poetry

# Install dependencies via Poetry
RUN poetry install --no-root

# Expose two folders outside of the container
VOLUME ["app/input", "app/output"]

# Optionally copy your project code
COPY . /app/

CMD poetry shell