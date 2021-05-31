SHELL = /bin/sh

.DEFAULT_GOAL := all

## help: Display list of commands
.PHONY: help
help: Makefile
	@sed -n 's|^##||p' $< | column -t -s ':' | sed -e 's|^| |'

## all: Run the main targets
.PHONY: all
all: install activate setup-run run-sequential report

## install: Install conda env.
.PHONY: install
install:
	conda env create --force -f environment.yml

## activate: Activate environment.
.PHONY: activate
activate:
	conda activate pdist_aggregation

## setup-run: Setup env variables for the run
.PHONY: setup-run
setup-run:
	export RUN_PREFIX=${date '+%Y-%m-%d-%H-%M-%S'}

## run-sequential: Run benchmarks for sequential mode (capped via taskset(1)).
.PHONY: run-sequential
run-sequential:
	taskset -c 0 python benchmarks/benchmark.py ${RUN_PREFIX}_seq

## run-parallel: Run benchmarks (default parallel execution)
.PHONY: run-parallel
run-parallel:
	python benchmarks/benchmark.py ${RUN_PREFIX}_par

## test: Launch all the test.
.PHONY: test
test:
	pytest tests

## report-sequential: Create a report for the sequential run
.PHONY: report-sequential
report-sequential:
	python benchmarks/report.py ${RUN_PREFIX}_seq
	pdfunite `ls *.pdf` ${RUN_PREFIX}_seq_results.pdf

## report-parallel: Create a report for the sequential run
.PHONY: report-parallel
report-parallel:
	python benchmarks/report.py ${RUN_PREFIX}_par
	pdfunite `ls *.pdf` ${RUN_PREFIX}_par_results.pdf
