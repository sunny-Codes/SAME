import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="same-pkg",
    version="1.0.0",
    author="Sunmin Lee",
    author_email="sunmin.lee@imo.snu.ac.kr",
    description="SAME: Skeleton-Agnostic Motion Embedding for Character Animation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
