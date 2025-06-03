# -*- coding: utf-8 -*-

# python setup.py sdist bdist_wheel
# python2 setup.py sdist bdist_wheel
# twine upload dist/*

import setuptools

with open("README.md") as f:
    README = f.read()

setuptools.setup(
    author="UCLA MII",
    name="superr_venn",
    license="MIT",
    description="superr_venn is an adapted tool for visualization of relations of many sets using matplotlib",
    version="0.0.1",
    long_description="See https://github.com/uclamii/superr_venn/blob/master/README.md",
    url="https://github.com/uclamii/superr_venn",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "matplotlib>=2.2.5", "pandas"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
