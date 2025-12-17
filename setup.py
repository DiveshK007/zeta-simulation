from setuptools import setup, find_packages

setup(
    name="zeta-simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "matplotlib>=3.8",
        "opencv-python>=4.8",
        "pybullet>=3.2",
        "torch>=2.0",
        "sentence-transformers>=3.0",
        "openai==0.28.1",
        "transformers>=4.30",
        "pytest>=7.4"
    ],
    python_requires=">=3.8",
    author="DiveshK007",
    description="ZETA Framework - Zero-shot Task Automation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="robotics ai zero-shot-learning",
)
