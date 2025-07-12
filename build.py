#!/usr/bin/env python3
"""
MSTML Framework - Full Build Script with Logging and Conda Setup
"""

import subprocess
import sys
import os
import shutil
import argparse
import platform
import logging
from typing import List, Tuple

class MStmlBuilder:
    def __init__(self, env_name: str = "mstml"):
        self.env_name = env_name
        self.has_conda = self._check_conda()

        # Setup logging
        logging.basicConfig(
            filename="setup.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.packages = [
            "python=3.10", "numpy", "pandas", "matplotlib", "networkx",
            "scipy", "scikit-learn", "gensim", "nltk", "cython",
            "setuptools", "wheel", "python-louvain", "notebook", "pip"
        ]
        self.pip_packages = ["pyldavis", "hypernetx", "mplcursors", "phate"]

    def _check_conda(self) -> bool:
        try:
            subprocess.run("conda --version", shell=True, check=True,
                           capture_output=True, text=True, timeout=10)
            return True
        except Exception:
            return False

    def _run_command(self, cmd: str, description: str) -> Tuple[bool, str, str]:
        print(f"\n{'='*60}\n{description}\n{'='*60}")
        logging.info(f"Running: {description}\nCommand: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True,
                                    capture_output=True, text=True, timeout=600)
            print("✓ Success!")
            logging.info(f"Success: {result.stdout.strip()}")
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e.stderr.strip()}")
            logging.error(f"Failed: {e.stderr.strip()}")
            return False, e.stdout or "", e.stderr or ""
        except subprocess.TimeoutExpired:
            print("✗ Timeout")
            logging.error("Command timed out")
            return False, "", "Command timed out"

    def _create_environment(self) -> bool:
        success, stdout, _ = self._run_command("conda env list", "Checking existing environments")
        if success and self.env_name in stdout:
            response = input(f"Environment '{self.env_name}' exists. Recreate it? (y/N): ").lower()
            if response == "y":
                self._run_command(f"conda env remove -n {self.env_name} -y",
                                  f"Removing existing environment {self.env_name}")
            else:
                return True
        cmd = f"conda create -n {self.env_name} -c conda-forge -c defaults {' '.join(self.packages)} -y"
        return self._run_command(cmd, f"Creating environment {self.env_name}")[0]

    def _install_pip_packages(self) -> bool:
        if not self.pip_packages:
            return True
        pip_cmd = f"conda run -n {self.env_name} pip install {' '.join(self.pip_packages)}"
        return self._run_command(pip_cmd, "Installing pip packages")[0]

    def _download_nltk_data(self) -> bool:
        nltk_data = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
        for data in nltk_data:
            cmd = f'conda run -n {self.env_name} python -c "import nltk; nltk.download(\\"{data}\\")"'
            self._run_command(cmd, f"Downloading NLTK: {data}")
        return True

    def _build_cython(self) -> bool:
        if not os.path.exists("setup.py"):
            logging.warning("setup.py not found, skipping Cython build")
            return True
        cmd = f"conda run -n {self.env_name} python setup.py build_ext --inplace"
        return self._run_command(cmd, "Compiling Cython extensions")[0]

    def _test_installation(self) -> bool:
        cmd = f'conda run -n {self.env_name} python -c "import mstml; print(\'✓ MSTML import successful\')"'
        return self._run_command(cmd, "Testing MSTML import")[0]

    def _print_summary(self):
        print(f"\n{'='*60}\nMSTML Setup Complete\n{'='*60}")
        print(f"Activate environment: conda activate {self.env_name}")
        print("Run: python -c \"import mstml; print('Ready!')\"")
        print("Deactivate: conda deactivate")

    def build(self) -> bool:
        print("MSTML Build Script")
        logging.info("Starting MSTML build process")

        if not self.has_conda:
            print("✗ Conda not found. Please install Miniconda manually.")
            logging.error("Conda not found.")
            return False

        if not self._create_environment():
            return False
        if not self._install_pip_packages():
            print("⚠ Some pip packages failed.")
        self._download_nltk_data()
        self._build_cython()
        if not self._test_installation():
            print("⚠ MSTML import test failed.")
        self._print_summary()
        return True

def main():
    parser = argparse.ArgumentParser(description="MSTML Framework Build Script")
    parser.add_argument("--env-name", default="mstml", help="Name for the conda environment")
    args = parser.parse_args()
    builder = MStmlBuilder(env_name=args.env_name)
    success = builder.build()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
