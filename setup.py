from setuptools import Command, find_packages, setup

__lib_name__ = "stSMILE"
__lib_version__ = "0.0.1"
__description__ = "SMILE is designed for multiscale dissection of spatial heterogeneity by integrating multi-slice spatial and single-cell transcriptomics."
__url__ = "https://github.com/zhanglhwhu/SMILE"
__author__ = "Lihua Zhang"
__author_email__ = "zhanglh@whu.edu.cn"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "data integration", "single cell", "deconvolution"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['stSMILE'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)
