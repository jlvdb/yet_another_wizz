import codecs
import setuptools
import os


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as f:
        return f.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


long_description = read("README.md")
scripts = [os.path.join("bin", fname) for fname in os.listdir("bin")]

setuptools.setup(
    name="yet_another_wizz",
    version=get_version("yet_another_wizz/__init__.py"),
    author="Jan Luca van den Busch",
    description="Implementation of the Schmidt et al. (2013) clustering redshift method and wrapper scripts to produce cross-correlation (CC) redshifts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlvdb/yet_another_wizz",
    packages=setuptools.find_packages(),
    scripts=scripts,
    install_requires=[
        "typing_extensions",
        "numpy",
        "pyarrow",
        "h5py",
        "pandas",
        "astropandas",
        "scipy",
        "astropy",
        "matplotlib"],
    extras_require={
        "treecorr": [
            "treecorr>=4.3"]}
)
