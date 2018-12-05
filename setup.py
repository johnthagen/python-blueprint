#!/usr/bin/env python3

import setuptools

setuptools.setup(
    name='fact',
    version='1.0.0',

    description='Example Python project',
    long_description=open('README.rst').read(),
    keywords=['python'],

    author='',
    url='https://github.com/johnthagen/python-blueprint',
    packages=setuptools.find_packages(),
    include_package_data=True,

    # This is a trick to avoid duplicating dependencies between both setup.py and
    #   requirements.txt. It does not work for all types of dependencies (e.g. VCS dependencies).
    #   requirements.txt must be included in MANIFEST.in for this to work.
    install_requires=open('requirements.txt').readlines(),
    zip_safe=False,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    entry_points={
        'console_scripts': ['fact=fact.cli:main'],
    }
)
