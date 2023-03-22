# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
import yaw

project = "yet_another_wizz"
copyright = "2023, Jan Luca van den Busch"
# The full version, including alpha/beta/rc tags.
release = yaw.__version__
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax"
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_inherit_docstrings = True
autosummary_generate = True
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["_static/custom.css"]
html_favicon = "_static/icon.ico"
html_theme_options = {
    "github_url": "https://github.com/jlvdb/yet_another_wizz",
    "collapse_navigation": True,
    "navigation_depth": 2,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navbar_align": "content",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "primary_sidebar_end": ["indices.html"],
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
   }
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"]
}
html_context = {
    "default_mode": "auto",
}

# -- Build custom files ------------------------------------------------------
from yaw.pipeline.default_setup import setup_default

with open("user_guide/cmd/default_setup.yaml", "w") as f:
    f.write(setup_default)

os.system("yaw --help > user_guide/yaw_help.txt")
for sub in (
    "init", "cross", "auto", "ztrue", "cache", "merge", "zcc", "plot", "run"
):
    os.system(f"yaw {sub} --help > user_guide/cmd/yaw_{sub}_help.txt")
