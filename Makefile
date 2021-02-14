.PHONY: deploy check-env
all: install
install: check-env clean
	ln -s $(PWD)/kosma_py $(GAG_PATH)/../pro/kosma_py
	ls *.class | xargs -I % ln -s $(PWD)/% $(GAG_PATH)/../pro/%
	ls doc/*.hlp | xargs -I % ln -s $(PWD)/% $(GAG_PATH)/../doc/hlp
	pip install -U -e kosma_py_lib

clean: check-env
	rm -f $(GAG_PATH)/../pro/kosma*.class
	if [ -d $(GAG_PATH)/../pro/kosma_py ]; then \
      rm -f $(GAG_PATH)/../pro/kosma_py; \
	fi
	rm -f $(GAG_PATH)/../doc/hlp/kosma*.hlp

check-env:
ifndef GAG_PATH
    $(error G_PATH is undefined please ensure GILDAS is installed and sourced)
endif

clean_test_output:
	rm -rf tests/data/*csv*
	rm -rf tests/spline_output/
	rm -rf tests/*lmv
	rm -rf tests/tmp*
	rm -rf tests/*figures*


tests = $(shell cd tests && ls *gildas)
run_tests:
	cd tests;\
	for test in $(tests); do\
		echo $${test}: testing against reference dataset; \
		class -nw @$${test}; \
	done
