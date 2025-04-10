from setuptools import setup, find_packages

setup(
    name="sheng-dholuo-translator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["pandas", "colorama", "fuzzywuzzy"],
    include_package_data=True,
    package_data={"sheng_dholuo_translator": ["phrases.csv"]},
    author="Kevin Omondi Jr.",
    author_email="kevojr69@gmail.com",
    description="A cultural nuance translator for Sheng and Dholuo",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KevinJr20/SDL-translator.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
