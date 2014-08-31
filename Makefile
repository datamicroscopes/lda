all:
	@echo "choose a valid target"

.PHONY: release
release:
	@echo "Setting up cmake (release)"
	@python ./cmake/print_cmake_command.py Release
	[ -d release ] || (mkdir release && cd release && eval `python ../cmake/print_cmake_command.py Release`)

.PHONY: relwithdebinfo
relwithdebinfo:
	@echo "Setting up cmake (relwithdebinfo)"
	@python ./cmake/print_cmake_command.py RelWithDebInfo
	[ -d relwithdebinfo ] || (mkdir relwithdebinfo && cd relwithdebinfo && eval `python ../cmake/print_cmake_command.py RelWithDebInfo`)

.PHONY: debug
debug:
	@echo "Setting up cmake (debug)"
	@python ./cmake/print_cmake_command.py Debug
	[ -d debug ] || (mkdir debug && cd debug && eval `python ../cmake/print_cmake_command.py Debug`)

.PHONY: test
test:
	(cd test && nosetests --verbose)

.PHONY: travis_install
travis_install:
	make relwithdebinfo
	(cd relwithdebinfo && make && make install)
	pip install .

.PHONY: travis_script
travis_script:
	(cd test && nosetests --verbose -a '!slow')

.PHONY: lint
lint:
	pyflakes microscopes test setup.py
	pep8 --filename=*.py --ignore=E265 microscopes test setup.py
	pep8 --filename=*.pyx --ignore=E265,E211,E225 microscopes

.PHONY: clean
clean:
	rm -rf release relwithdebinfo debug microscopes_lda.egg-info
	find microscopes/ -name '*.cpp' -type f -print0 | xargs -0 rm --
	find microscopes/ -name '*.so' -type f -print0 | xargs -0 rm --
	find microscopes/ -name '*.pyc' -type f -print0 | xargs -0 rm --
