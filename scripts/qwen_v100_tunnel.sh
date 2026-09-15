#!/usr/bin/env bash
# Runs on viewing; both forwarding endpoints bind only to loopback.
set -u
while true; do
  ssh -N -o BatchMode=yes -o StrictHostKeyChecking=yes \
    -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:18000:127.0.0.1:8001 george@george-server
  sleep 15
done
