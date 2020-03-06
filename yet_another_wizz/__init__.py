import importlib
if importlib.util.find_spec("pyarrow") is None:
    raise ImportError("package 'pyarrow' not found")
