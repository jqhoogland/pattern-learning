from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="patterns",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="A brief description of your project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/anonymous",  # Replace with your project's repository URL
    packages=find_packages(
        include=["patterns", "patterns.*"]
    ),  # This includes all sub-packages inside `patterns`
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Modify if you have a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Modify if you have a different minimal Python version requirement
)
