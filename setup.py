from setuptools import setup, find_packages
setup(
    name="quest-videoqa",
    packages=find_packages(where=".", include=["src*"]),
)
