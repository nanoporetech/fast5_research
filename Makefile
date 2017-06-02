.PHONY: install test develop deb docs clean

SHELL=/bin/bash
PKG_NAME=fast5_research
# We could create custom docker images for CI builds with the venv pre-installed
# to /root/venv when the docker images is build so our CI builds run fast. To
# allow us to use the same Makefile locally, assume that if /root/venv exists,
# we are on a docker image, so don't need to create a venv, else we are local
# and need to. 
VENV_DIR=$(shell eval 'if [ -d /root/venv ]; then echo /root/venv; else echo `pwd`/venv/${PKG_NAME}; fi')
VENV=${VENV_DIR}/bin/activate
TRUST=''
EXTRA_URL=''
START_DIR=$(shell pwd)

venv:
	# if venv exists, do nothing, else create venv and install all dependencies
	test -d ${VENV_DIR} || virtualenv -p python2 ${VENV_DIR}; \
							. ${VENV}; \
							pip install --upgrade pip; \
							pip install -r dev_requires.txt ${TRUST} ${EXTRA_URL}; \
							pip install -r requirements.txt ${TRUST} ${EXTRA_URL}; 



install: venv clean
	. ${VENV} && python setup.py install

test: develop
	. ${VENV} && nosetests

develop: clean
	. ${VENV} && python setup.py develop 

sdist: clean
	. ${VENV} && python setup.py sdist

test_sdist:
	virtualenv test_sdist && . test_sdist/bin/activate && pip install --upgrade pip && pip install -f ./dist/ $(PKG_NAME)

deb: clean
	. ${VENV} && python setup.py --command-packages=stdeb.command sdist_dsc --debian-version 1~$(VERSION_TAG) bdist_deb

docs:
	. ${VENV} && sphinx-apidoc --no-toc -o docs/api $(PKG_NAME)
	. ${VENV} && python setup.py build_sphinx

clean: venv
	. ${VENV} && python setup.py clean
	rm -rf dist/ deb_dist/ *.egg-info/
	find ./${PKG_NAME} -name '*.pyc' -delete
	find ./${PKG_NAME} -name '*.so' -delete
