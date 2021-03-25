from setuptools import find_packages, setup

setup(
    name=’AuDoLab’,
    packages=find_packages(include=[`AuDoLab´]),
    version=’0.1.0',
    description=’My first Python library’,
    author=’Arne Tillmann’,
    license=’GNU General Public License v3.0’,
    install_requires=[],
    setup_requires=[‘pytest-runner’],
    tests_require=[‘pytest==4.4.1’],
    test_suite=’tests’,
)
