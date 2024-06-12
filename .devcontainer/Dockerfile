# Define build arguments
ARG USERNAME=nonroot

FROM python:3.10-slim-bullseye

# Set environment variables
ENV HOME=/home/$USERNAME

# Use build arguments
ARG USERNAME

# Install necessary packages for autocomplete and enhanced prompt
RUN apt-get update && apt-get install -y \
    bash-completion \
    less \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create a nonroot user
RUN groupadd --gid 1000 $USERNAME; \ 
    useradd --uid 1000 --gid 1000 -m $USERNAME; \
    true

# Ensure nonroot user has the correct home directory permissions
RUN chown -R $USERNAME:$USERNAME $HOME

## Add pip location to PATH
ENV PATH="${HOME}/.local/bin:${PATH}"

# Set bash as the default shell for the nonroot user
RUN chsh -s /bin/bash $USERNAME

# Switch to nonroot user
USER $USERNAME

# Upgrade pip
RUN pip install --upgrade pip

# Install production dependencies
RUN pip install tensorflow==2.15.0
RUN pip install matplotlib

# Set working directory
WORKDIR /workspace
