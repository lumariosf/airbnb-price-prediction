import os
import subprocess
import venv

def create_virtualenv(env_name="venv"):
    venv.create(env_name, with_pip=True)
    print("Virtual environment created.")

def install_packages(env_name="venv"):
    if os.name == "nt":  # Windows
        pip_path = os.path.join(env_name, "Scripts", "pip.exe")
    else: 
        pip_path = os.path.join(env_name, "bin", "pip")

    packages = ["jupyter", "pandas", "scikit-learn", "seaborn", "statsmodels"]

    subprocess.check_call([pip_path, "install"] + packages)


if __name__ == "__main__":
    create_virtualenv()
    install_packages()
