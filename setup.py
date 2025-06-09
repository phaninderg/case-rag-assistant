from setuptools import setup, find_packages

setup(
    name="case_rag_assistant",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.9",
)
