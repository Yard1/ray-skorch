import io
import os
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

setup(
    name="ray-skorch",
    packages=find_packages(),
    version="0.0.1",
    author="Antoni Baum & contributors",
    author_email="antoni.baum@protonmail.com",
    description=("Skorch on Ray Train"),
    long_description=io.open(
        os.path.join(ROOT_DIR, "README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=open("./requirements.txt").read(),
)
