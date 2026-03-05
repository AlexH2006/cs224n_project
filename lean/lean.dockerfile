FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl git build-essential

# Install elan + stable Lean toolchain (provides lake)
ENV ELAN_HOME="/root/.elan"
ENV PATH="${ELAN_HOME}/bin:${PATH}"
RUN curl https://elan.lean-lang.org/elan-init.sh -sSf | bash -s -- --default-toolchain leanprover/lean4:stable -y

# Create Lean/Mathlib project at /lean-checker
WORKDIR /
RUN lake new lean-checker math
WORKDIR /lean-checker
RUN lake exe cache get
