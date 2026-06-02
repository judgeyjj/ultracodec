"""Setup script for the UltraCodec package."""
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

REQUIREMENTS = []
req_file = ROOT / "requirements.txt"
if req_file.exists():
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            REQUIREMENTS.append(line)


setup(
    name="ultracodec",
    version="0.1.0",
    description="UltraCodec: an ultra-low frame-rate (<5 Hz) neural speech codec for LLMs.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="UltraCodec Project",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(include=["ultracodec", "ultracodec.*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ultracodec-train=scripts.train:main",
            "ultracodec-eval=scripts.evaluate:main",
        ],
    },
)
