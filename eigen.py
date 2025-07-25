import os
import urllib.request
import tarfile
from pathlib import Path

eig_ver = "3.4.0"
eig_url = f"https://gitlab.com/libeigen/eigen/-/archive/{eig_ver}/eigen-{eig_ver}.tar.gz"
eig_dir = Path("eigen_download")

eig_common_paths = [
    Path("/usr/include"),
    Path("/usr/include/eigen3"),
    Path("/usr/local/include"),
    Path("/usr/local/include/eigen3"),
    Path("/usr/local/include/eigen3"),
    Path("C:/Program Files/Eigen/include/eigen3"),
    Path("C:/Eigen/include/eigen3"),
]

def find_system_eigen():

    env_path = os.getenv("EIGEN_DIR")
    if env_path:
        env_path = Path(env_path)
        if (env_path / "Eigen" / "Dense").exists():
            print(f"Found Eigen in EIGEN_DIR env var: {env_path}")
            return env_path   

    for p in eig_common_paths:
            print(f"Found Eigen in system path: {p}")
            return p

    return None

def download_eigen():

    existing_path = find_system_eigen()
    if existing_path:
        print(f"Using existing Eigen headers at {existing_path}")
        return existing_path

    if eig_dir.exists():
        print(f"Eigen directory {eig_dir} already exists, skipping download.")
        return eig_dir
    else:
        os.mkdir(eig_dir)

    print(f"Downloading Eigen {eig_ver}...")
    archive_path = f"eigen-{eig_ver}.tar.gz"
    urllib.request.urlretrieve(eig_url, archive_path)

    print("Extracting Eigen...")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "Eigen" in m.name or "unsupported" in m.name]
        tar.extractall(path=eig_dir.parent, members=members)

    extracted_folder = eig_dir.parent / f"eigen-{eig_ver}"
    if extracted_folder.exists():
        extracted_folder.rename(eig_dir)

    os.remove(archive_path)
    print("Eigen downloaded and ready.")
    return eig_dir

