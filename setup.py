from setuptools import setup, find_packages

from pathlib import Path

# プロジェクトのルートディレクトリ
ROOT_DIR = Path(__file__).parent
README = ROOT_DIR / "README.md"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
VERSION = "1.0.0"

# READMEの内容を読み込み
long_description = ""
if README.exists():
    with open(README, 'r', encoding='utf-8') as f:
        long_description = f.read()

# requirements.txtの内容を読み込み
install_requires = []
if REQUIREMENTS.exists():
    with open(REQUIREMENTS, 'r', encoding='utf-8') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="tumblr-image-collector-simple",
    version=VERSION,
    author="Simple Tools Developer",
    author_email="developer@example.com",
    description="Simple and practical Tumblr image collection tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["tumblr_image_collector_simple", "tumblr_image_collector_simple.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "tumblr-collector=tumblr_collector_simple:main",
        ],
    },
    keywords="tumblr image collection simple practical",
    license="MIT",
    zip_safe=False,
)
