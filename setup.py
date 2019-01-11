import os
import re
import sys
from setuptools import setup, find_packages


__pkg_name__ = 'fast5_research'
__author__ = 'cwright'
__description__ = 'ONT Research .fast5 read/write API.'

# Use readme as long description and say its github-flavour markdown
from os import path
this_directory = path.abspath(path.dirname(__file__))
kwargs = {'encoding':'utf-8'} if sys.version_info.major == 3 else {}
with open(path.join(this_directory, 'README.md'), **kwargs) as f:
    __long_description__ = f.read()
__long_description_content_type__ = 'text/markdown'


# Get the version number from __init__.py
verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))

dir_path = os.path.dirname(__file__)
with open(os.path.join(dir_path, 'requirements.txt')) as fh:
    install_requires = [
        r.split('#')[0].strip()
        for r in fh.read().splitlines() if not r.strip().startswith('#')
    ]

extra_requires={}

py2only_requirements = ['futures']
if len(py2only_requirements) > 0:
    extra_requires[':python_version == "2.7"'] = []

for requirement in py2only_requirements:
    install_requires.remove(requirement)
    extra_requires[':python_version == "2.7"'].append(requirement)


setup(
    name=__pkg_name__,
    version=__version__,
    url='https://github.com/nanoporetech/{}'.format(__pkg_name__),
    author=__author__,
    author_email='{}@nanoporetech.com'.format(__author__),
    description=__description__,
    long_description=__long_description__,
    long_description_content_type=__long_description_content_type__,
    entry_points={
        'console_scripts': [
            'extract_reads = {}.extract:extract_reads'.format(__pkg_name__),
            'filter_reads = {}.extract:filter_multi_reads'.format(__pkg_name__),
        ]
    },
    license='Mozilla Public License 2.0',
    dependency_links=[],
    install_requires=install_requires,
    tests_require=['nose>=1.3.7'].extend(install_requires),
    extras_require=extra_requires,
    packages=find_packages(exclude=['*.test', '*.test.*', 'test.*', 'test']),
    package_data={},
    zip_safe=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords='ONT Research fast5 API',
)
