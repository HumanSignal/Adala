"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import re
import setuptools

# Module dependencies
requirements, dependency_links = [], []
with open('requirements.txt') as f:
    for line in f.read().splitlines():
        requirements.append(line)

setuptools.setup(
    name='adala',
    version='0.0.2',
    author='Heartex',
    author_email="hello@humansignal.com",
    description='ADALA: Automated Data Labeling Agent',
    url='https://github.com/HumanSignal/ADALA',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements
)
