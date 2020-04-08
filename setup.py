"""Setup file
"""

import setuptools

import gravtools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='gravtools',
                 version=gravtools.__version__,
                 description='Gravitational tools for Python',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url=gravtools.__github_url__,
                 author='James W. Kennington',
                 author_email='jameswkennington@gmail.com',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
