import importlib.util
import io
import logging
import os
import sys
from typing import List

from setuptools import find_packages, setup


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    return _read_requirements("requirements.txt")


setup(
    name="etalon",
    version="0.1.0",
    author="etalon Team",
    license="Apache 2.0",
    description="A framework for benchmarking LLM Inference Systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/project-etalon/etalon",
    project_urls={
        "Homepage": "https://github.com/project-etalon/etalon",
        "Documentation": "project-etalon.readthedocs.io",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("docs")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "vllm": ["vllm"]
    },
)