import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-YOUR-USERNAME-HERE", # Replace with your own username
    version="0.0.0a1",
    author="Averell Gatton",
    author_email="gattonian@gmail.com",
    description="A package for Deep Dreaming with the 3D IM-NET model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuantumPlumber/DeepDream3D.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='>=3.6',
)