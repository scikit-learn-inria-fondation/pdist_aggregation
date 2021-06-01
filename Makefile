SHELL = /bin/bash
PROJECT = pdist_aggregation

VENV_PATH=`conda info --base`/envs/${PROJECT}
PYTHON_EXECUTABLE=${VENV_PATH}/bin/python
PYTEST_EXECUTABLE=${VENV_PATH}/bin/pytest

.DEFAULT_GOAL := all

## help: Display list of commands
.PHONY: help
help: Makefile
	@sed -n 's|^##||p' $< | column -t -s ':' | sed -e 's|^| |'

## all: Run the main targets
.PHONY: all
all: install benchmark-sequential benchmark-parallel

## install: Install conda env.
.PHONY: install
install:
	conda env create --force -f environment.yml

## benchmark-sequential: Run benchmarks for sequential execution, 'NAME' variable can be provided
# Uses taskset to cap to a cpu solely
.PHONY: benchmark-sequential
benchmark-sequential:
		@[ "${NAME}" ] || export NAME=comp
		taskset -c 0 ${PYTHON_EXECUTABLE} benchmarks/benchmark.py ${NAME}seq

## benchmark-parallel: Run benchmarks for parallel execution, 'NAME' variable can be provided
.PHONY: benchmark-parallel
benchmark-parallel:
		@[ "${NAME}" ] || export NAME=comp
		${PYTHON_EXECUTABLE} benchmarks/benchmark.py ${NAME}par

## test: Launch all the test.
.PHONY: test
test:
	${PYTEST_EXECUTABLE} tests
