from setuptools import setup, find_packages

setup(
    name="sheng-dholuo-translator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "colorama",
        "fuzzywuzzy",
        "transformers",
        "torch",
        "datasets",
        "sentencepiece",
    ],
    include_package_data=True,
    package_data={
        "sheng_dholuo_translator": ["phrases.csv"]
    },
    author="Kevin Ochieng Omondi",
    author_email="your.email@example.com",
    description="A cultural nuance translator for Sheng and Dholuo with AI support",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KevinJr20/SDL-translator.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "sheng-dholuo = sheng_dholuo_translator.translator:main",
        ],
    },
)