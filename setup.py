import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beast",
    version="0.0.1",
    author="Chris Walsh",
    author_email="c.walsh@beatson.gla.ac.uk",
    description="Provides whole slide stain normalisation and augmentation utilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cwalshai.github.io/beatson-augmentation-stain-normalisation-toolkit-docs/",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "imagecodecs",
        "numpy",
        "opencv-python-headless",
        "openslide-python",
        "Pillow",
        "scipy",
        "spams",
        "tifffile"
    ]
)
