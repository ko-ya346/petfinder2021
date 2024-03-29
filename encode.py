import base64
import gzip
import re
from pathlib import Path

from src.misc import get_current_commit_hash

template = """
import gzip
import base64
import os
from pathlib import Path
from typing import Dict
# this is base64 encoded source code
file_data: Dict = {file_data}
for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))
def run(command):
    os.system('echo "from setuptools import setup; setup(name=\\'src\\', packages=[\\'src\\'],)" > setup.py')
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)
run('python setup.py develop --install-dir /kaggle/working')
# output current commit hash
print('{commit_hash}')
"""


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode("utf-8")


def build_script():
    to_encode = [
        path
        for path in Path("./").glob("src/**/*")
        if re.search("/*\.(py|yaml)", str(path))
    ]
    #    to_encode.extend(list(Path("./").glob('output/ex001/*.ckpt')))
    file_data = {str(path): encode_file(path) for path in to_encode}
    output_path = Path(".build/script.py")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        template.replace("{file_data}", str(file_data)).replace(
            "{commit_hash}", get_current_commit_hash()
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    build_script()
