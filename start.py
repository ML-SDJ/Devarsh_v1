"""Quick environment check for the MedExplain AI project.

Run ``python start.py`` after cloning the repository to verify the
Python dependencies, dataset folders, and fine-tuned checkpoints that the
pipelines expect. The script prints guidance for any missing pieces so
you can resolve them before launching the individual modules or the
Streamlit app.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from textwrap import indent

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "torchcam",
    "streamlit",
    "pillow",
    "numpy",
    "transformers",
    "datasets",
    "evaluate",
    "rouge_score",
]

DATA_FOLDERS = [
    Path("data/chest_xray/train"),
    Path("data/chest_xray/val"),
    Path("data/chest_xray/test"),
]

ARTIFACT_PATHS = {
    "vision": Path("artifacts/vision/resnet18_finetuned.pth"),
    "nlp": Path("artifacts/nlp/t5_small_finetuned"),
}


def _highlight(title: str) -> str:
    bar = "=" * len(title)
    return f"{title}\n{bar}"


def check_dependencies() -> list[str]:
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return missing


def check_dataset() -> list[Path]:
    missing: list[Path] = []
    for folder in DATA_FOLDERS:
        if not folder.exists() or not any(folder.glob("**/*")):
            missing.append(folder)
    return missing


def check_artifacts() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for key, path in ARTIFACT_PATHS.items():
        if path.is_dir():
            ready = any(path.rglob("*"))
        else:
            ready = path.is_file()
        status[key] = ready
    return status


def main() -> None:
    project_root = Path(__file__).parent
    print(_highlight("MedExplain AI environment report"))
    print(f"Python {sys.version.split()[0]}")
    print(f"Project root: {project_root.resolve()}\n")

    missing_packages = check_dependencies()
    if missing_packages:
        print(_highlight("Missing Python packages"))
        formatted = "\n".join(f"- {pkg}" for pkg in missing_packages)
        print(f"The following required packages were not found:\n{formatted}")
        print(
            "Install them with: pip install -r requirements.txt\n"
            "(Use the CUDA variant of torch/torchvision if you have a GPU.)\n"
        )
    else:
        print("All required Python packages are available.\n")

    missing_data = check_dataset()
    if missing_data:
        print(_highlight("Dataset folders not ready"))
        notes = "\n".join(
            f"- Expected '{folder}' with the Kaggle Chest X-Ray Pneumonia data"
            for folder in missing_data
        )
        instructions = (
            "Download the dataset from Kaggle and extract the 'train', 'val', and '\n"
            "'test' folders into data/chest_xray/."
        )
        print(f"{notes}\n{instructions}\n")
    else:
        print("Chest X-ray dataset folders look good.\n")

    artifact_status = check_artifacts()
    print(_highlight("Model artifacts"))
    lines = []
    for key, ready in artifact_status.items():
        if ready:
            lines.append(f"- {key}: ready")
        else:
            if key == "vision":
                remedy = (
                    "Run vision_pipeline.train_model() and save_model() to create "
                    "artifacts/vision/resnet18_finetuned.pth."
                )
            else:
                remedy = (
                    "Run nlp_pipeline.fine_tune(...) to populate "
                    "artifacts/nlp/t5_small_finetuned."
                )
            lines.append(f"- {key}: missing — {remedy}")
    print(indent("\n".join(lines), ""))

    print(
        "\nNext steps:\n"
        "1. Resolve any missing packages with 'pip install -r requirements.txt'.\n"
        "2. Populate the Kaggle dataset folders under data/chest_xray/.\n"
        "3. Train the models via vision_pipeline.py and nlp_pipeline.py to generate artifacts.\n"
        "4. Launch the UI with 'streamlit run streamlit_app.py'."
    )


if __name__ == "__main__":
    main()
