import io
import tokenize

import black


def remove_comments(source_code):
    """Remove all comments from Python source code."""
    tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)
    new_tokens = [token for token in tokens if token.type != tokenize.COMMENT]
    return tokenize.untokenize(new_tokens).strip()


def reformat_code(source_code):
    return black.format_str(source_code, mode=black.Mode())


def process_python_code(source_code):
    stripped_code = remove_comments(source_code)
    formatted_code = reformat_code(stripped_code)
    return formatted_code.replace(r"\n\n\n", r"\n\n")


def protect_main(source_code):
    lines = source_code.split("\n")

    last_import = -1
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import = i

    source_code = "\n".join(lines[: last_import + 1])
    source_code += '\n\nif __name__ == "__main__":\n'
    source_code += "\n    ".join(lines[last_import + 1 :])
    return source_code


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    with open(path) as f:
        source = f.read()

    clean_code = process_python_code(source)
    script = protect_main(clean_code)
    print(script)
