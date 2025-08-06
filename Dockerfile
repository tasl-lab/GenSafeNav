# Step 1: Use the official PyTorch image as a base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Step 2: Set the working directory in the container
WORKDIR /workspace

# Step 3: Install system dependencies with fixed versions
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Install Python packages
RUN pip install --no-cache-dir \
    pandas \
    gym \
    scipy \
    opencv-python \
    matplotlib \
    Cython \
    imageio \
    wandb \
    imageio[ffmpeg]

# Step 5: Copy and install Python-RVO2
COPY Python-RVO2 /tmp/Python-RVO2
RUN if [ -d /tmp/Python-RVO2/build ]; then rm -rf /tmp/Python-RVO2/build; fi \
    && cd /tmp/Python-RVO2 \
    && python setup.py build \
    && python setup.py install \
    && rm -rf /tmp/Python-RVO2

# # Step 6: Copy your application code to the container
# COPY . /workspace

# Step 7: Create a user with the same UID and GID as the host user
ARG USER_ID
ARG GROUP_ID
RUN groupadd -g ${GROUP_ID} dockeruser && \
    useradd -u ${USER_ID} -g dockeruser -m dockeruser && \
    chown -R dockeruser:dockeruser /workspace

# Step 8: Set environment variables
ENV PYTHONUNBUFFERED=1

# Step 9: Expose a port if your application uses one
EXPOSE 8888

# Step 10: Switch to the new user
USER dockeruser

# Step 11: Specify the command to run on container start
CMD ["bash"]
