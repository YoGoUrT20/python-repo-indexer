from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repo-indexer",
    version="0.1.0",
    author="YoGoUrT20",
    description="Repository indexing and context search system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YoGoUrT20/python-repo-indexer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sqlalchemy>=2.0.0",
        "sqlite3-api>=1.0.0",
        "sentence-transformers>=2.2.2",
        "langchain>=0.1.0",
        "tqdm>=4.66.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pygit2>=1.12.0",
    ],
    entry_points={
        "console_scripts": [
            "repo-indexer=src.__main__:main",
        ],
    },
) 