# Build image

```bash
podman build -f Dockerfile -t lsaie_project:base .
```

# Save built image
```bash
enroot import -x mount -o /iopsstor/scratch/cscs/$USER/images/lsaie_project.sqsh podman://localhost/lsaie_project:base
```

# Run container with
```bash
srun -A a-infra02 -p normal --time=1:00:00 --environment=/users/$USER/LSAIE-Project/docker/lsaie_project.toml --pty bash
```

# Give proper read permissions to everyone (optional)
```bash
chmod o+x /iopsstor/scratch/cscs/$USER/images
chmod o+r /iopsstor/scratch/cscs/$USER/images/lsaie_project.sqsh
```
