#syntax=docker/dockerfile:1
FROM python:3.10.16-slim-bookworm AS build
ENV PYTHONDONTWRITEBYTECODE=1
RUN ["/usr/local/bin/python3.10", "-m", "venv", "/venv"]
RUN ["/venv/bin/pip3", "install", "--disable-pip-version-check", "--upgrade", "pip", "setuptools", "wheel"]


FROM build AS build-venv
COPY ["./requirements.txt", "."]
ENV PYTHONDONTWRITEBYTECODE=1
RUN ["/venv/bin/pip", "install", "-r", "./requirements.txt", "--no-cache-dir", "--extra-index-url=https://download.pytorch.org/whl/cpu"]
RUN ["mkdir", "/app"]
RUN ["mkdir", "/app/assets"]

FROM gcr.io/distroless/base-debian12:nonroot

COPY --from=build-venv ["/usr/local/bin/python3.10", "/usr/local/bin/python3.10"]
COPY --from=build-venv ["/usr/local/lib/python3.10", "/usr/local/lib/python3.10"]
COPY --from=build-venv ["/usr/local/lib/libpython3.10.so.1.0", "/usr/lib/"]
COPY --from=build-venv ["/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
COPY --chown=nonroot:nonroot --from=build-venv ["/venv/", "/home/nonroot/venv/"]
COPY --chown=nonroot:nonroot --from=build-venv ["/app", "/home/nonroot/app"]
COPY --chown=nonroot:nonroot ["./app.py", "./util.py", "./opt.py", "./mylog.py", "./expr_parser.py", "/home/nonroot/app"]
COPY --chown=nonroot:nonroot ["./assets", "/home/nonroot/app/assets"]

ENV OMP_NUM_THREADS=4
WORKDIR "/home/nonroot/app"

ARG GPBO_WEBAPP_VERSION
ENV GPBO_WEBAPP_VERSION=${GPBO_WEBAPP_VERSION}

CMD ["/home/nonroot/venv/bin/python3.10", "app.py"]
