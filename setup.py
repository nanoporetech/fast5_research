import os
import re
from setuptools import setup, find_packages


__pkg_name__ = 'fast5_research'

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

extra_requires = {
}

setup(
    name=__pkg_name__,
    version=__version__,
    url='https://github.com/nanoporetech/{}'.format(__pkg_name__),
    author='mwykes',
    author_email='mwykes@nanoporetech.com',
    description='ONT Research fast5 read/write package',
    license='Mozilla Public License 2.0',
    dependency_links=[],
    install_requires=install_requires,
    tests_require=['nose>=1.3.7'].extend(install_requires),
    extras_require=extra_requires,
    packages=find_packages(exclude=['*.test', '*.test.*', 'test.*', 'test']),
    package_data={},
    zip_safe=True,
    test_suite='discover_tests',
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
