from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="uetasr",
    version="0.0.1",
    description="UETASR: A TensorFlow package for speech recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="thanhtvt & others",
    author_email="trantrongthanhhp@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    url="https://github.com/thanhtvt/uetasr"
)
