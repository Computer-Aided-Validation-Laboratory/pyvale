import pathlib
from setuptools import  setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import os
from glob import glob
import pybind11
import urllib.request
import tarfile

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")



debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')  # Remove so setup() doesn't get confused

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'


eig_ver = "3.4.0"
eig_url = f"https://gitlab.com/libeigen/eigen/-/archive/{eig_ver}/eigen-{eig_ver}.tar.gz"
eig_dir = pathlib.Path("eigen_download")

eig_common_paths = [
    pathlib.Path("/usr/include"),
    pathlib.Path("/usr/include/eigen3"),
    pathlib.Path("/usr/local/include"),
    pathlib.Path("/usr/local/include/eigen3"),
    pathlib.Path("/usr/local/include/eigen3"),
    pathlib.Path("C:/Program Files/Eigen/include/eigen3"),
    pathlib.Path("C:/Eigen/include/eigen3"),
]

def find_system_eigen():

    env_path = os.getenv("EIGEN_DIR")
    if env_path:
        env_path = pathlib.Path(env_path)
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



ext_cython = Extension(
        "pyvale.cython.rastercyth",
        ["src/pyvale/cython/rastercyth.py",],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-ffast-math",openmp_arg],
        extra_link_args=[openmp_arg],
    )

ext_dic = Extension(
    'pyvale.dic.dic2dcpp',
    sorted(glob("src/pyvale/dic/cpp/dic*.cpp")),
    language="c++",
    include_dirs=[pybind11.get_include()],
    extra_compile_args=['-g', '-O0', '-fopenmp'] if debug_mode else ['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp', '-lfftw3'] + (['-g'] if debug_mode else []),
)
ext = cythonize([ext_cython], annotate=True) + [ext_dic]

setup(
      ext_modules=ext,
)
