FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    sudo \
    bzip2 \
    libx11-6 

RUN apt install -y build-essential \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    gcc \
    python3-dev \
    ffmpeg \
    portaudio19-dev 
# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="/root/.pyenv/bin:$PATH"
RUN eval "$(pyenv init --path)"
RUN pyenv install 3.10.9 && pyenv global 3.10.9


RUN python3 --version

WORKDIR /app
# COPY
COPY ./ /app/

# Python environment settings
ENV POETRY_HOME="/root/.local" 
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Setup the virtual environment
RUN poetry config virtualenvs.in-project true
RUN poetry install

# Start shell
CMD ["/bin/bash"]