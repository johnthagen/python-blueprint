# Use a multi-stage build to reduce the size of the final image.
#   This example is optimized to reduce final image size rather than for simplicity.
# Using a -slim image also greatly reduces image size.
# It is possible to use -alpine images instead to further reduce image size, but this comes
# with several important caveats.
#   - Alpine images use MUSL rather than GLIBC (as used in the default Debian-based images).
#   - Most Python packages that require C code are tested against GLIBC, so there could be
#     subtle errors when using MUSL.
#   - These Python packages usually only provide binary wheels for GLIBC, so the packages
#     will need to be recompiled fully within the container images, increasing build times.
FROM python:3.13-slim-trixie AS python_builder

# Pin uv to a specific version to make container builds reproducible.
ENV UV_VERSION=0.9.26
ENV UV_PYTHON_DOWNLOADS=never

# Set ENV variables that make Python more friendly to running inside a container.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

# By default, pip caches copies of downloaded packages from PyPI. These are not useful within
# a contain image, so disable this to reduce the size of images.
ENV PIP_NO_CACHE_DIR=1
ENV WORKDIR=/src

WORKDIR ${WORKDIR}

# Install any system dependencies required to build wheels, such as C compilers or system packages
# For example:
#RUN apt-get update && apt-get install -y \
#    gcc \
#    && rm -rf /var/lib/apt/lists/*

# Install uv into the global environment to isolate it from the venv it creates.
RUN pip install "uv==${UV_VERSION}"

# Pre-download/compile wheel dependencies into a virtual environment.
# Doing this in a multi-stage build allows omitting compile dependencies from the final image.
# This must be the same path that is used in the final image as the virtual environment has
# absolute symlinks in it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Copy in project dependency specification.
COPY pyproject.toml uv.lock LICENSE.txt ./

# Install only project dependencies, as this is cached until pyproject.toml uv.lock are updated.
RUN uv sync --locked --no-default-groups --no-install-project

# Copy in source files.
# README.md is required for the package to build. It can be ommited for non-package applications.
COPY README.md ./
COPY src src

# Install the rest of the application into the virtual environment.
# Omit this step if your project is a non-package application and copy the source in the second
# stage instead.
RUN uv sync --locked --no-default-groups --no-editable

## Final Image
# The image used in the final image MUST match exactly to the python_builder image.
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ENV HOME=/home/user
ENV APP_HOME=${HOME}/app

# Create the home directory for the new user.
RUN mkdir -p ${HOME}

# Create the user so the program doesn't run as root. This increases security of the container.
RUN groupadd -r user && \
    useradd -r -g user -d ${HOME} -s /sbin/nologin -c "Container image user" user

# Setup application install directory.
RUN mkdir ${APP_HOME}

# If you use Docker Compose volumes, you might need to create the directories in the image,
# otherwise when Docker Compose creates them they are owned by the root user and are inaccessible
# by the non-root user. See https://github.com/docker/compose/issues/3270

WORKDIR ${APP_HOME}

# Copy and activate pre-built virtual environment.
COPY --from=python_builder ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# For non-package applications, COPY source files here rather than in the --no-editable step
# in the python_builder stage.

# Give access to the entire home folder to the new user so that files and folders can be written
# there. Some packages such as matplotlib, want to write to the home folder.
RUN chown -R user:user ${HOME}

ENTRYPOINT ["fact"]
