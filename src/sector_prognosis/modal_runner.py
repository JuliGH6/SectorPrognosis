"""Optional Modal GPU runner for large-scale neural architecture & Optuna sweeps.

Usage (requires Modal API token: `modal token set --token-id ... --token-secret ...`):
    uv pip install modal
    modal run src/sector_prognosis/modal_runner.py
"""

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("sector-prognosis-gpu")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "scikit-learn",
            "lightgbm",
            "xgboost",
            "catboost",
            "torch",
            "optuna",
        )
        .add_local_dir("src", remote_path="/root/src")
        .add_local_file("data/merged_clean.parquet", remote_path="/root/data/merged_clean.parquet")
    )

    @app.function(image=image, gpu="T4", timeout=1800)
    def run_gpu_sweep():
        import os
        import sys
        import torch

        sys.path.insert(0, "/root/src")
        os.chdir("/root")
        print(f"Running on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        from sector_prognosis.models_v2 import run_iteration_4

        run_iteration_4()

    @app.local_entrypoint()
    def main():
        run_gpu_sweep.remote()
