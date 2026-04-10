# Isaac Sim 6.0 Setup Guide

Tested and verified on:
- **GPU**: NVIDIA RTX 5090 (32GB VRAM, Blackwell, compute capability 12.0)
- **Driver**: 595.58.03
- **OS**: Ubuntu 22.04.5 LTS (x86_64)
- **CPU**: AMD Ryzen Threadripper 7960X 24-core
- **RAM**: 128GB

This setup uses **Isaac Sim 6.0.0.0** (pip) and **Isaac Lab v3.0.0-beta** (source) to support newer NVIDIA drivers (590+) that are incompatible with Isaac Sim 5.1.

## Prerequisites

- Python 3.12 (installed automatically by uv)
- [uv](https://github.com/astral-sh/uv) package manager
- GCC/G++ 11 (Ubuntu 22.04 default)

## 1. Clone and set up sim-evals

```bash
git clone git@github.com:arhanjain/sim-evals.git
cd sim-evals
```

### Clone Isaac Lab v3.0.0-beta

```bash
git clone --depth 1 -b v3.0.0-beta https://github.com/isaac-sim/IsaacLab.git submodules/IsaacLab
```

### Install Python 3.12 and sync dependencies

```bash
uv python install 3.12
uv lock
uv sync
```

### Apply Isaac Lab device fix

Isaac Lab v3.0.0-beta has a bug where `ArticulationView.get_*()` returns CPU arrays when the simulation runs on GPU. Two files need patching:

**`submodules/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py`**

Add after `self._root_view` assignment in `__init__`:
```python
def _ensure_device(self, arr):
    """Ensure a warp array is on the correct device, cloning if needed."""
    import warp as wp
    if arr.device != self.device:
        return wp.clone(arr, device=self.device)
    return arr
```

Then wrap all `self._root_view.get_*()` direct assignments (that don't already use `wp.clone(..., device=self.device)`) with `self._ensure_device()`. For example:
```python
# Before:
self._joint_pos.data = self._root_view.get_dof_positions()
# After:
self._joint_pos.data = self._ensure_device(self._root_view.get_dof_positions())
```

Apply the same pattern to:
- `get_dof_positions()`, `get_dof_velocities()`
- `get_root_transforms()`, `get_root_velocities()`
- `get_link_transforms()`, `get_link_velocities()`, `get_link_accelerations()`
- `get_coms()`, `get_masses()`, `get_inertias()`
- `get_link_incoming_joint_force()`
- `get_dof_limits()` (the `.assign()` call)

**`submodules/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object_data.py`**

Same pattern — add `_ensure_device` helper and wrap:
- `get_transforms()`, `get_velocities()`, `get_accelerations()`, `get_coms()`

### Download simulation assets

```bash
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
```

## 2. Set up openpi (policy server)

In a separate directory:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
```

### Fix for Blackwell GPUs

The bundled `ptxas` (CUDA 12.6) doesn't support Blackwell (CC 12.0). Edit `uv.lock` to bump `nvidia-cuda-nvcc-cu12` from `12.6.85` to `12.9.86`:

Find the `[[package]]` block for `nvidia-cuda-nvcc-cu12` and replace the version and wheel URL:
```toml
name = "nvidia-cuda-nvcc-cu12"
version = "12.9.86"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/25/48/b54a06168a2190572a312bfe4ce443687773eb61367ced31e064953dd2f7/nvidia_cuda_nvcc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl", hash = "sha256:5d6a0d32fdc7ea39917c20065614ae93add6f577d840233237ff08e9a38f58f0", size = 40546229 },
]
```

Then upgrade JAX for Blackwell bf16 support:
```bash
uv sync
uv pip install "jax[cuda12]==0.5.3"
```

**Important**: Use `uv run --no-sync` to prevent uv from reverting JAX.

## 3. Running evaluations

### Terminal 1: Policy server (from openpi/)

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 uv run --no-sync scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_droid_jointpos \
  --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
```

Wait for `server listening on 0.0.0.0:8000`.

### Terminal 2: Simulator (from sim-evals/)

Headless (saves video to `runs/`):
```bash
cd sim-evals
OMNI_KIT_ACCEPT_EULA=yes uv run --no-sync python run_eval.py --headless --episodes 3 --scene 1
```

Scenes: `1` = put cube in bowl, `2` = put can in mug, `3` = put banana in bin.

Videos are saved to `runs/<date>/<time>/episode_*.mp4`.

## Known issues

- **Headful Kit viewport**: The Isaac Sim GUI opens but does not render robot meshes. This appears to be an Isaac Sim 6.0 early developer preview limitation on Blackwell GPUs. Headless mode with video recording works correctly.
- **opencv-python (non-headless)**: Conflicts with Isaac Sim's bundled GTK libraries, crashing the GPU foundation plugin. Must use `opencv-python-headless`.
- **inotify warnings**: Harmless spam. Fix with: `sudo sysctl fs.inotify.max_user_watches=524288`
- **`uv run` vs `uv run --no-sync`**: Always use `--no-sync` for both openpi and sim-evals to prevent uv from reverting manually upgraded packages (JAX, nvcc).

## Version summary

| Component | Version |
|---|---|
| Isaac Sim | 6.0.0.0 (pip, NVIDIA PyPI) |
| Isaac Lab | v3.0.0-beta (source) |
| Python | 3.12 |
| PyTorch | 2.10.0 (CUDA 12.8, pulled by isaacsim) |
| JAX (openpi) | 0.5.3 |
| Warp | 1.12.0 |
| nvidia-cuda-nvcc | 12.9.86 |
