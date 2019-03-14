###
# This Makefile is simply for testing and making docs, to install
# the project it should be sufficient to use python setup.py <cmd>


.PHONY: docs clean test test_py2 test_py3


venv_py2/bin/activate:
	test -d venv_py2 || virtualenv venv_py2 --prompt '(fast5_py2) ' --python=python2
	. $@ && pip install pip --upgrade
	. $@ && pip install -r dev_requirements.txt 
	. $@ && pip install -r requirements.txt; 

test_py2: venv_py2/bin/activate
	. $< && python setup.py nosetests


venv_py3/bin/activate:
	test -d venv_py3 || virtualenv venv_py3 --prompt '(fast5_py3) ' --python=python3
	. $@ && pip install pip --upgrade
	. $@ && pip install -r dev_requirements.txt 
	. $@ && pip install -r requirements.txt; 

test_py3: venv_py3/bin/activate
	. $< && python setup.py nosetests


test: test_py2 test_py3

clean:
	rm -rf build dist *.egg-info venv_* 

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

DOCSRC = docs

docs: venv_py3/bin/activate
	. $< && pip install sphinx sphinx_rtd_theme sphinx-argparse
	. $< && cd $(DOCSRC) && $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	rm -rf docs/modules.rst docs/fast5_research.rst
	@echo
	@echo "Build finished. The HTML pages are in $(DOCSRC)/$(BUILDDIR)/html."
	touch $(DOCSRC)/$(BUILDDIR)/html/.nojekyll
