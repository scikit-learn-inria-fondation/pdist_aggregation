SHELL = /bin/bash
PROJECT = pdist_aggregation

VENV_PATH=`conda info --base`/envs/${PROJECT}
PYTHON_EXECUTABLE=${VENV_PATH}/bin/python
PYTEST_EXECUTABLE=${VENV_PATH}/bin/pytest
JUPYTER_EXECUTABLE=${VENV_PATH}/bin/jupyter
COMMIT=`git rev-parse --short HEAD`

.DEFAULT_GOAL := all

## help: Display list of commands
.PHONY: help
help: Makefile
	@sed -n 's|^##||p' $< | column -t -s ':' | sed -e 's|^| |'

## all: Run the main targets
.PHONY: all
all: install benchmark-parallel benchmark-sequential report-parallel report-sequential

## install: Install conda env.
.PHONY: install
install:
	conda env create --force -f environment.yml

## benchmark-sequential: Run benchmarks for sequential execution, 'NAME' variable can be provided
# Uses taskset to cap to a cpu solely
.PHONY: benchmark-sequential
benchmark-sequential:
		@[ "${NAME}" ] || export NAME=${COMMIT}
		taskset -c 0 ${PYTHON_EXECUTABLE} benchmarks/benchmark.py ${NAME}seq

## benchmark-parallel: Run benchmarks for parallel execution, 'NAME' variable can be provided
.PHONY: benchmark-parallel
benchmark-parallel:
		@[ "${NAME}" ] || export NAME=${COMMIT}
		${PYTHON_EXECUTABLE} benchmarks/benchmark.py ${NAME}par

## report-sequential: Report benchmarks for sequential execution, 'NAME' variable can be provided
.PHONY: report-sequential
report-parallel:
		@[ "${NAME}" ] || export NAME=${COMMIT}
		${PYTHON_EXECUTABLE} benchmarks/report.py ${NAME}seq

## report-parallel: Report benchmarks for parallel execution, 'NAME' variable can be provided
.PHONY: report-parallel
report-parallel:
		@[ "${NAME}" ] || export NAME=${COMMIT}
		${PYTHON_EXECUTABLE} benchmarks/report.py ${NAME}par

.PHONY: notebook
notebook:
		NAME=${COMMIT} ${JUPYTER_EXECUTABLE} nbconvert --to html --execute --output benchmarks/results/index.html visualization.ipynb

## test: Launch all the test.
.PHONY: test
test:
	${PYTEST_EXECUTABLE} tests
