from distutils.core import setup

setup(
    name = "SMILE",
    packages = ['SMILE']
    version = "0.1",
    license = 'MIT'
    description = "Multiscale dissection of spatial heterogeneity by integrating multi-slice spatial and single-cell transcriptomics",
    author = "Lihua Zhang",
    author_email = "zhanglh@whu.edu.cn",
    url = 'https://github.com/lhzhanglabtools/SMILE/archive/master.zip',
    keywords = ["spatial transcriptomics", "data integration", "scRNA-seq", "spatial domain", "deconvolution"],
    install_requires = ["requests",]
    zip_safe = False,
    include_package_data = True,
)
