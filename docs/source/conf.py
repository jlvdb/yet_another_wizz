# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

PKG_ROOT = Path("../../").resolve()

try:  # user has installed the package
    import yaw
except ImportError:  # try local package location
    import sys

    sys.path.insert(0, str(PKG_ROOT / "src"))

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
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_inherit_docstrings = True
autodoc_typehints = "signature"
autosummary_generate = True
autoclass_content = "class"
autodoc_default_options = {
    "exclude-members": "__init__",
}

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
    "pygments_light_style": "xcode",
    "pygments_dark_style": "github-dark",
    "announcement": f"<p>Version 3.0 released!</p>",
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}
html_context = {
    "default_mode": "auto",
}

# -- Build custom files ------------------------------------------------------

def write_changes(path):
    version_key = "Version"
    version_info = dict()

    with (PKG_ROOT / "CHANGELOG.rst").open() as f:
        version = None

        lines = f.readlines()
        for start, line in enumerate(lines):
            if line.startswith(version_key):
                break

        for line in lines[start:]:
            if line.startswith("---"):
                continue

            elif line.startswith(version_key):
                _, version = line.strip().split()
                version_info[version] = []

            else:
                version_info[version].append(line)

    with open(path, "w") as f:
        f.write("Change log\n==========\n\n")

        for i, (verion, info_lines) in enumerate(version_info.items()):
            f.write(f".. dropdown:: Version {verion}\n")
            f.write("    :class-title: h5\n")
            if i == 0:
                f.write("    :open:\n")

            for line in info_lines:
                f.write(f"    {line}")


def write_readme(path):
    path = Path(path).resolve()
    path.parent.mkdir(exist_ok=True)

    print(f"generating '{path}'")
    with (
        (PKG_ROOT / "README.rst").open() as source,
        path.open("w") as f,
    ):
        lines = source.readlines()

        for i, line in enumerate(lines):
            if "end header" in line:
                start = i + 1
                break
        else:
            raise ValueError("missing 'end header' comment")

        f.write("..\n")
        f.write("    This is a copy of /README.rst with its header stripped.\n")
        f.write("\n")

        f.writelines(lines[start:])


write_readme("user_guide/README.rst")
write_changes("changes.rst")
