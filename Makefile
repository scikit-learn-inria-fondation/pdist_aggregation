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

## benchmark-sequential: Run benchmarks for sequential mode (capped via taskset(1)), 'NAME' variable has to be provided needed
.PHONY: benchmark-sequential
benchmark-sequential: activate
		@[ "${NAME}" ] || ( echo ">> NAME must be set with 'make command NAME=value'"; exit 1 )
		taskset -c 0 python benchmarks/benchmark.py ${NAME}_seq

## benchmark-parallel: Run benchmarks (default parallel execution), 'NAME' variable has to be provided needed
.PHONY: benchmark-parallel
benchmark-parallel: activate
		@[ "${NAME}" ] || ( echo ">> NAME must be set with 'make command NAME=value'"; exit 1 )
		python benchmarks/benchmark.py ${NAME}_par

## test: Launch all the test.
.PHONY: test
test: activate
	pytest tests
