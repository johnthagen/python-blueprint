# Use a multi-stage build to reduce the size of the final image.
#   This example is optimized to reduce final image size rather than for simplicity.
# Using a -slim also greatly reduced image size.
# It is possible to use -alpine images instead to further reduce image size, but this comes
# with several important caveats.
#   - Alpine images use MUSL rather than GLIBC (as used in the default Debian-based images).
#   - Most Python packages that require C code are tested against GLIBC, so there could be
#     subtle errors when using MUSL.
#   - These Python packages usually only provide binary wheels for GLIBC, so the packages
#     will need to be recompiled fully within the Docker images, increasing build times.
FROM python:3.8-slim-buster AS python_builder

# Set ENV variables that make Python more friendly to running inside a container.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

# By default, pip caches copies of downloaded packages from PyPI. These are not useful within
# a Docker image, so disable this to reduce the size of images.
ENV PIP_NO_CACHE_DIR 1
ENV WORKDIR /src

# This must be the same path that is used in the final image as the virtual environment has
# absoulte symlinks in it.
ENV VIRTUAL_ENV /opt/venv

WORKDIR ${WORKDIR}

# Install any system depdendencies required to build wheels, such as C compilers or system packages
# For example:
#RUN apt-get update && apt-get install -y \
#    gcc \
#    && rm -rf /var/lib/apt/lists/*

# Pre-download/compile wheel dependencies into a virtual environment.
# Doing this in a multi-stage build allows ommitting compile dependencies from the final image.
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements.txt ${WORKDIR}
RUN pip install --upgrade pip wheel && \
    pip install -r requirements.txt

# Copy in source files.
COPY LICENSE.txt MANIFEST.in pyproject.toml README.rst requirements.txt setup.py ./
COPY src src

# Install console script.
RUN pip install .

## Final Image
# The image used in the final image MUST match exactly to the python_builder image.
FROM python:3.8-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV VIRTUAL_ENV /opt/venv

ENV HOME /home/user
ENV APP_HOME ${HOME}/app

# Create the home directory for the new user.
RUN mkdir -p ${HOME}

# Create the user so the program doesn't run as root. This increases security of the container.
RUN groupadd -r user && \
    useradd -r -g user -d ${HOME} -s /sbin/nologin -c "Docker image user" user

# Setup application install directory.
RUN mkdir ${APP_HOME}

WORKDIR ${APP_HOME}

# Copy and activate pre-built virtual environment.
COPY --from=python_builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:${PATH}"

# For Python applications that are not installable libraries, you may need to copy in source
# files here in the final image rather than in the python_builder image.

# Give access to the entire home folder to the new user so that files and folders can be written
# there. Some packages such as matplotlib, want to write to the home folder.
RUN chown -R user:user ${HOME}

ENTRYPOINT ["fact"]
