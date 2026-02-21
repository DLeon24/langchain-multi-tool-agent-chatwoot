# Dockerfile for LangChain/FastAPI Backend

# 1. Base image: python:3.11-slim is an excellent choice, lightweight and modern.
FROM python:3.11-slim

# 2. Set working directory in container
WORKDIR /app

# 3. Copy requirements file first to leverage Docker cache.
# This means if you only change your Python code but not dependencies,
# Docker won't need to reinstall everything every time you build the image.
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code.
# This includes your 'src' folder, the '.env' file if you have it locally (though better to handle as secret), etc.
COPY . /app

# 6. Expose port used by Uvicorn. We will use 8000 as standard for APIs.
EXPOSE 8000

# 7. Command to run the application in production.
# This is the command that will run when the container starts.
CMD ["uvicorn", "main_chatwoot_bot:app", "--host", "0.0.0.0", "--port", "8000"]