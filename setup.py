#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

setup_requirements = ['pytest-runner', 'numpy==1.19.2', 'pandas==1.2.3', 'webbot==0.34', 'bs4==0.0.1', 'gensim==3.8.3', 'pyldavis==2.1.2', 'scikit-learn==0.24.1', 'keras==2.3.1', 'nltk==3.5', 'lime==0.2.0.1', 'matplotlib==3.3.4']

test_requirements = ['pytest>=3', ]

setup(
    author="Arne Tillmann",
    author_email='arne.tillmann.vellmar@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=" With AuDoLab you can perform Latend Direchlet Allocation on highly imbalanced datasets.",
    entry_points={
        'console_scripts': [
            'AuDoLab=AuDoLab.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='AuDoLab',
    name='AuDoLab',
    packages=find_packages(include=['AuDoLab', 'AuDoLab.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ArneTillmann/AuDoLab',
    version='0.0.35',
    zip_safe=False,
)
