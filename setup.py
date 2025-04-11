import os
import re
import setuptools


def read_version():
    # importing gpustat causes an ImportError :-)
    __PATH__ = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(__PATH__, 'modelhub_client_trt/__init__.py')) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")


# get project version
__version__ = read_version()

# get project long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# get project requirements list
with open("requirements.txt", "r", encoding="utf-8") as fh:
    packages = fh.read().split("/n")

setuptools.setup(
    name="modelhub-client-trt",
    version=__version__,
    author='RIA.com',
    author_email='oleg.cherniy@ria.com',
    description="RIA ModelHub with TensorRT support tools package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ria-com/modelhub-client-trt.git",
    packages=setuptools.find_packages(),
    install_requires=packages,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
    keywords='modelhub modelhub-client modelhub-client-trt TensorRT ria-com ria.com ria',
    python_requires='>=3.9'
)
