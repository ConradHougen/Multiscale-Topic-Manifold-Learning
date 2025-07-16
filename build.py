#!/usr/bin/env python3
"""
MSTML Framework - One-Time Environment Setup and Install
"""

import subprocess
import sys
import os
import argparse
import logging

class MstmlBuilder:
    def __init__(self, env_name: str = "mstml"):
        self.env_name = env_name
        self.has_conda = self._check_conda()
        self.conda_packages = self._load_requirements("conda_requirements.txt")
        self.pip_requirements_file = "pip_requirements.txt"

        # Setup logging
        logging.basicConfig(
            filename="build.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _check_conda(self) -> bool:
        try:
            env_name = os.environ.get("CONDA_DEFAULT_ENV")
            if env_name:
                print(f"Detected conda env: {env_name}")
                return True
            else:
                return False
        except:
            return False

    def _load_requirements(self, filename):
        if not os.path.exists(filename):
            return []
        with open(filename, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def _run(self, cmd: str, desc: str):
        print(f"\n{'='*60}\n{desc}\n{'='*60}")
        logging.info(f"Running: {desc}\nCommand: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("Success")
            logging.info(result.stdout.strip())
            return True
        except subprocess.CalledProcessError as e:
            print("Failed:", e.stderr.strip())
            logging.error(e.stderr.strip())
            return False

    def _create_or_reuse_env(self):
        success, result = self._run("conda env list", "Checking for existing environments"), ""
        if not success:
            return False
        result = subprocess.run("conda env list", shell=True, capture_output=True, text=True).stdout
        if self.env_name in result:
            resp = input(f"Environment '{self.env_name}' exists. Recreate it? (y/N): ").lower()
            if resp == "y":
                self._run(f"conda env remove -n {self.env_name} -y", f"Removing {self.env_name}")
            else:
                return True
        conda_cmd = f"conda create -n {self.env_name} -c conda-forge -c defaults {' '.join(self.conda_packages)} -y"
        return self._run(conda_cmd, f"Creating environment {self.env_name}")

    def _install_pip_and_package(self):
        pip_cmd = f"conda run -n {self.env_name} pip install -e ."
        if os.path.exists(self.pip_requirements_file):
            pip_cmd += f" -r {self.pip_requirements_file}"
        return self._run(pip_cmd, "Installing pip packages and MSTML in editable mode")

    def _download_nltk(self):
        nltk_data = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
        for pkg in nltk_data:
            cmd = f'conda run -n {self.env_name} python -c "import nltk; nltk.download(\'{pkg}\')"'
            self._run(cmd, f"Downloading NLTK: {pkg}")
        return True

    def _test_import(self):
        return self._run(
            f'conda run -n {self.env_name} python -c "import mstml; print(\'✓ MSTML import successful\')"',
            "Testing mstml import"
        )

    def build(self):
        print("Starting MSTML build...")
        if not self.has_conda:
            print("Conda not found. Please install Miniconda or Anaconda.")
            return False
        if not self._create_or_reuse_env():
            return False
        if not self._install_pip_and_package():
            return False
        self._download_nltk()
        if not self._test_import():
            print("Import test failed")
        print(f"\n{'='*60}\nMSTML Setup Complete\n{'='*60}")
        print(f"Activate with: conda activate {self.env_name}")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="mstml", help="Name of the conda environment to create/use")
    args = parser.parse_args()
    success = MstmlBuilder(env_name=args.env_name).build()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
