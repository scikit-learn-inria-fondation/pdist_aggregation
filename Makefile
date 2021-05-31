SHELL = /bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE_CMD=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

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

## activate: Activate environment.
.PHONY: activate
activate:
	$(CONDA_ACTIVATE_CMD) pdist_aggregation
	@echo "Python executable: `which python`"

## benchmark-sequential: Run benchmarks for sequential execution, 'NAME' variable can be provided
# Uses taskset to cap to a cpu solely
.PHONY: benchmark-sequential
benchmark-sequential: activate
		@[ "${NAME}" ] || export NAME=comp
		taskset -c 0 python benchmarks/benchmark.py ${NAME}_seq

## benchmark-parallel: Run benchmarks for parallel execution, 'NAME' variable can be provided
.PHONY: benchmark-parallel
benchmark-parallel: activate
		@[ "${NAME}" ] || export NAME=comp
		python benchmarks/benchmark.py ${NAME}_par

## test: Launch all the test.
.PHONY: test
test: activate
	pytest tests
