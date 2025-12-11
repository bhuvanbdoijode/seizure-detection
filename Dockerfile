FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV (Debian Trixie compatible)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 10000

# Start Flask with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
