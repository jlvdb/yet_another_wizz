import setuptools
import os


with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    install_requires = [pkg.strip() for pkg in f.readlines() if pkg.strip()]
scripts = [
    os.path.join("bin", fname) for fname in os.listdir("bin")]

setuptools.setup(
    name="yet_another_wizz",
    version="1.1",
    author="Jan Luca van den Busch",
    description="Implementation of the Schmidt et al. (2013) clustering redshift method and wrapper scripts to produce cross-correlation (CC) redshifts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlvdb/yet_another_wizz",
    packages=setuptools.find_packages(),
    scripts=scripts,
    install_requires=install_requires)
