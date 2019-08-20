from setuptools import setup, find_packages
from os import path as os_path

this_directory = os_path.abspath(os_path.dirname(__file__))

def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(
    name = 'SKNet',
    version = '0.0.1',
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    description = 'a library used for stacking based on scikit-learn',
    license = 'MIT License',
    url = 'https://github.com/zhangruochi/SKNet',
    author = 'Ruochi Zhang',
    author_email = 'zrc720@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires= read_requirements('requirements.txt'),
    keywords=['stack', 'sklearn'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)