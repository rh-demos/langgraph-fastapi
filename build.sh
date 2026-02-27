#!/usr/bin/env bash

podman build --arch amd64 --os linux -t quay.io/jonkey/langgraph-fastapi:1.0.4 -f Dockerfile .
podman push quay.io/jonkey/langgraph-fastapi:1.0.4
