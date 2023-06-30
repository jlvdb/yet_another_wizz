# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
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
    "sphinx_design",
    "sphinx_copybutton",
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

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_line_continuation_character = "\\"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/icon.ico"
html_theme_options = {
    "github_url": "https://github.com/jlvdb/yet_another_wizz",
    "collapse_navigation": True,
    "navigation_depth": 3,
    "show_nav_level": 3,
    "show_toc_level": 3,
    "navbar_align": "content",
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "pygment_light_style": "xcode",
    "pygment_dark_style": "github-dark"
}
html_sidebars = {
    "**": [
        "search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}
html_context = {
    "default_mode": "auto",
}

# -- Build custom files ------------------------------------------------------

# generate a changelog file with dropdown sections
ver_strs = []
ver_texts = []
with open("../../CHANGELOG.rst") as f:
    for line in f.readlines():
        if line.startswith("Version"):
            if len(ver_strs) > 0:
                ver_texts.append(ver_text)
            ver_strs.append(line.strip().split()[1])
            ver_text = []
        elif len(ver_strs) > 0 and not line.startswith("---"):
            ver_text.append(line)
    else:
        ver_texts.append(ver_text)
with open("changes.rst", "w") as f:
    f.write("Change log\n==========\n\n")
    for i, (ver_str, ver_text) in enumerate(zip(ver_strs, ver_texts)):
        f.write(f".. dropdown:: Version {ver_str}\n")
        f.write(f"    :class-title: h5\n")
        if i == 0:
            f.write(f"    :open:\n")
        for line in ver_text:
            f.write(f"    {line}")

path = "user_guide/README.rst"
if not os.path.exists(path):
    print(f"generating '{path}'")
    with open(f"../../{os.path.basename(path)}") as r:
        with open(path, "w") as f:
            lines = r.readlines()
            for line in lines[2:]:
                f.write(line)

path = "user_guide/cmd/default_setup.yaml"
if not os.path.exists(path):
    print(f"generating '{path}'")
    from yaw.pipeline.default_setup import setup_default

    with open(path, "w") as f:
        f.write(setup_default)

for sub in (
    "", "init", "cross", "auto", "ztrue", "cache", "merge", "zcc", "plot", "run"
):
    path = f"user_guide/cmd/yaw_help_{sub}.txt"
    if not os.path.exists(path):
        print(f"generating '{path}'")
        os.system(f"yaw {sub} --help > {path}")
