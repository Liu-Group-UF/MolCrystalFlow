from setuptools import setup, find_packages

setup(
    name="molcrystalflow",
    version="0.1.0",
    packages=find_packages(include=["molcrystalflow", "molcrystalflow.*"]),
    author="Cheng Zeng",
    description="Molecular Crystal Structure Prediction via Flow Matching",
    python_requires=">=3.10",
)
