#!/usr/bin/env python3
"""
MSTML Framework - One-Time Environment Setup and Install

This build script automatically strips ANSI escape sequences from conda output
while preserving real-time progress display and creating clean log files.
"""

import subprocess
import sys
import os
import argparse
import logging
import re

class MstmlBuilder:
    def __init__(self, env_name: str = "mstml", force_cpu_only: bool = False):
        self.env_name = env_name
        self.force_cpu_only = force_cpu_only
        self.has_conda = self._check_conda()
        all_packages = self._load_requirements("conda_requirements.txt")
        
        # Separate FAISS packages (need special channel) from regular packages
        self.faiss_packages = [pkg for pkg in all_packages if 'faiss' in pkg.lower()]
        self.conda_packages = [pkg for pkg in all_packages if 'faiss' not in pkg.lower()]
        
        # Force CPU-only if requested
        if self.force_cpu_only and self.faiss_packages:
            self.faiss_packages = [pkg.replace('faiss-gpu', 'faiss-cpu') for pkg in self.faiss_packages]
            print(f"⚠ Forced CPU-only mode: FAISS packages changed to {self.faiss_packages}")
        
        self.pip_requirements_file = "requirements.txt"

        # Setup logging
        logging.basicConfig(
            filename="build.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _strip_ansi_escape_sequences(self, text: str) -> str:
        """
        Remove ANSI escape sequences from text while preserving content.
        
        This removes terminal control characters (like cursor movements, colors)
        that conda uses for progress displays but keeps the actual progress info.
        """
        # Pattern to match ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

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

    def _run(self, cmd: str, desc: str, show_progress: bool = True):
        print(f"\n{'='*60}\n{desc}\n{'='*60}")
        logging.info(f"Running: {desc}\nCommand: {cmd}")
        
        if show_progress and ("conda create" in cmd or "conda install" in cmd):
            # For conda commands, show real-time progress while capturing output
            return self._run_with_progress(cmd, desc)
        else:
            # For other commands, use simple capture
            return self._run_simple(cmd, desc)
    
    def _run_simple(self, cmd: str, desc: str):
        """Run command with simple output capture."""
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("Success")
            
            # Strip ANSI escape sequences from output before logging
            clean_stdout = self._strip_ansi_escape_sequences(result.stdout.strip())
            if clean_stdout:
                logging.info(clean_stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("Failed:", e.stderr.strip())
            
            # Strip ANSI escape sequences from error output before logging
            clean_stderr = self._strip_ansi_escape_sequences(e.stderr.strip())
            if clean_stderr:
                logging.error(clean_stderr)
            return False
    
    def _run_with_progress(self, cmd: str, desc: str):
        """Run command while showing real-time progress and capturing clean logs."""
        try:
            # Start the process
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            output_lines = []
            # Read output line by line and display progress
            for line in process.stdout:
                # Show progress to user (with ANSI sequences for live display)
                print(line, end='')
                
                # Store clean version for logging
                clean_line = self._strip_ansi_escape_sequences(line.strip())
                if clean_line:  # Only store non-empty lines
                    output_lines.append(clean_line)
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                print("\nSuccess")
                # Log the clean output
                if output_lines:
                    logging.info('\n'.join(output_lines))
                return True
            else:
                print(f"\nFailed with return code: {process.returncode}")
                if output_lines:
                    logging.error('\n'.join(output_lines))
                return False
                
        except Exception as e:
            print(f"Failed: {str(e)}")
            logging.error(f"Command failed: {str(e)}")
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

    def _get_platform(self):
        """Detect the current platform."""
        import platform
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "darwin":
            return "osx"  
        elif system == "windows":
            return "windows"
        else:
            return "unknown"

    def _install_faiss(self):
        """Install FAISS using conda packages with proper dependency resolution."""
        if not self.faiss_packages:
            print("No FAISS packages to install")
            return True
            
        print(f"\n{'='*60}\nFAISS Installation\n{'='*60}")
        
        platform = self._get_platform()
        print(f"Detected platform: {platform}")
        
        # Determine if we should attempt GPU installation
        should_try_gpu = False
        gpu_detected = False
        
        if platform == "linux" and not self.force_cpu_only:
            # Try to detect GPU but don't let detection failure prevent GPU installation attempt
            try:
                result = subprocess.run("nvidia-smi", capture_output=True)
                gpu_detected = (result.returncode == 0)
                if gpu_detected:
                    print("✓ NVIDIA GPU detected on Linux")
                else:
                    print("⚠ nvidia-smi not found, but will still try FAISS-GPU (may work anyway)")
            except FileNotFoundError:
                print("⚠ nvidia-smi not found, but will still try FAISS-GPU (may work anyway)")
            
            # Always attempt GPU on Linux unless user explicitly said --cpu-only
            should_try_gpu = True
        elif self.force_cpu_only:
            print("⚠ CPU-only mode forced by user (--cpu-only flag)")
        else:
            print(f"⚠ FAISS-GPU not available on {platform} (Linux only)")
        
        # Install based on platform and user preference
        if should_try_gpu:
            # Linux with GPU: try cuVS first, then regular GPU, then CPU fallback
            print("Installing FAISS with GPU support...")
            
            # Try cuVS version (most advanced)
            cuvs_cmd = f"conda install -n {self.env_name} -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs -y"
            if self._run(cuvs_cmd, "Installing FAISS-GPU with cuVS (most advanced)"):
                test_cmd = f'conda run -n {self.env_name} python -c "import faiss; print(f\'FAISS ready with {{faiss.get_num_gpus()}} GPU(s)\')"'
                if self._run(test_cmd, "Testing FAISS-GPU-cuVS", show_progress=False):
                    print("✓ FAISS-GPU with cuVS installed successfully!")
                    return True
                else:
                    print("cuVS installation failed, trying regular GPU version...")
                    self._run(f"conda remove -n {self.env_name} faiss-gpu-cuvs libnvjitlink -y", "Cleaning up cuVS")
            
            # Try regular GPU version
            gpu_cmd = f"conda install -n {self.env_name} -c pytorch -c nvidia faiss-gpu -y"
            if self._run(gpu_cmd, "Installing FAISS-GPU (regular)"):
                test_cmd = f'conda run -n {self.env_name} python -c "import faiss; print(f\'FAISS ready with {{faiss.get_num_gpus()}} GPU(s)\')"'
                if self._run(test_cmd, "Testing FAISS-GPU", show_progress=False):
                    print("✓ FAISS-GPU installed successfully!")
                    return True
                else:
                    print("GPU version failed, falling back to CPU...")
                    self._run(f"conda remove -n {self.env_name} faiss-gpu -y", "Cleaning up GPU version")
        
        # Install CPU version (Linux without GPU, Windows, OSX, or fallback)
        print("Installing FAISS-CPU...")
        
        # Step 1: Install MKL dependencies for better compatibility (especially on Windows/Linux)
        print("Installing MKL dependencies for FAISS...")
        mkl_cmd = f"conda install -n {self.env_name} -c conda-forge mkl -y"
        self._run(mkl_cmd, "Installing MKL math libraries", show_progress=False)
        
        # Step 2: Try conda-forge first (better dependency resolution)
        print("Trying conda-forge FAISS-CPU (recommended)...")
        forge_cmd = f"conda install -n {self.env_name} -c conda-forge faiss-cpu -y"
        if self._run(forge_cmd, "Installing FAISS-CPU from conda-forge"):
            test_cmd = f'conda run -n {self.env_name} python -c "import faiss; print(\'FAISS version:\', faiss.__version__); print(\'FAISS-CPU ready\')"'
            if self._run(test_cmd, "Testing conda-forge FAISS-CPU", show_progress=False):
                print("✓ FAISS-CPU (conda-forge) installed successfully!")
                return True
            else:
                print("conda-forge version has import issues, trying PyTorch channel...")
                self._run(f"conda remove -n {self.env_name} faiss faiss-cpu libfaiss -y", "Cleaning up conda-forge version")
        
        # Step 3: Try PyTorch channel as fallback
        print("Trying PyTorch channel FAISS-CPU...")
        pytorch_cmd = f"conda install -n {self.env_name} -c pytorch faiss-cpu -y"
        if self._run(pytorch_cmd, "Installing FAISS-CPU from PyTorch channel"):
            test_cmd = f'conda run -n {self.env_name} python -c "import faiss; print(\'FAISS version:\', faiss.__version__); print(\'FAISS-CPU ready\')"'
            if self._run(test_cmd, "Testing PyTorch FAISS-CPU", show_progress=False):
                print("✓ FAISS-CPU (PyTorch) installed successfully!")
                return True
            else:
                print("PyTorch version has import issues...")
                self._run(f"conda remove -n {self.env_name} faiss-cpu libfaiss -y", "Cleaning up PyTorch version")
        
        print("❌ FAISS installation failed. MSTML will use scipy fallbacks.")
        print("This is normal and doesn't affect functionality - scipy provides the same results.")
        return False

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
        # Test MSTML import
        mstml_success = self._run(
            f'conda run -n {self.env_name} python -c "import mstml; print(\'✓ MSTML import successful\')"',
            "Testing mstml import"
        )
        
        # Test FAISS import (optional) - final verification
        print("\n" + "="*60 + "\nFinal FAISS Status Check\n" + "="*60)
        faiss_test = f'conda run -n {self.env_name} python -c "import faiss; print(\'FAISS ready for use\')"'
        if self._run(faiss_test, "Final FAISS verification", show_progress=False):
            print("✓ FAISS is available and ready")
        else:
            print("⚠ FAISS not available - MSTML will use scipy fallbacks")
        
        return mstml_success

    def build(self):
        print("Starting MSTML build...")
        if not self.has_conda:
            print("Conda not found. Please install Miniconda or Anaconda.")
            return False
        if not self._create_or_reuse_env():
            return False
        if not self._install_faiss():
            print("Warning: FAISS installation failed. Continuing without FAISS support.")
        if not self._install_pip_and_package():
            return False
        self._download_nltk()
        if not self._test_import():
            print("Import test failed")
        print(f"\n{'='*60}\nMSTML Setup Complete\n{'='*60}")
        print(f"Activate with: conda activate {self.env_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description="MSTML Framework Build Script")
    parser.add_argument("--env-name", default="mstml", help="Name of the conda environment to create/use")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only installation (no GPU acceleration)")
    args = parser.parse_args()
    
    success = MstmlBuilder(env_name=args.env_name, force_cpu_only=args.cpu_only).build()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
