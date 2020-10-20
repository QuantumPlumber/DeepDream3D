# this dockerfile merely adds the latest updates to the base build. Build dockerfile base and then run this.

FROM deepdream3d:1.0

# Set the working directory.
WORKDIR /usr/DeepDream3D

# Copy project over
COPY . .

ENTRYPOINT ["./docker_entrypoint.sh"]