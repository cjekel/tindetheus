#/bin/make
#@ Tindetheus

TINDETHEUS_NAME := "Tindetheus"
TINDETHEUS_DESCRIPTION := "Personalized machine learning models for Tinder."
TINDETHEUS_VERSION := "v0.5.5"
TINDETHEUS_ROOT := ${PWD}
SHELL := /bin/bash
PATH := "${TINDETHEUS_ROOT}/.venv/bin:${PATH}"

%: %-tindetheus
	@true

.PHONY: help-tindetheus
help-tindetheus:  ## Display this help.
	@cd ${TINDETHEUS_ROOT} && awk 'BEGIN {FS = ":.*##"; print ""${TINDETHEUS_NAME}" "${TINDETHEUS_VERSION}"\n"${TINDETHEUS_DESCRIPTION}"\n\nUsage: make \033[36m<command>\033[0m\n\nCommands:"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: init-tindetheus
init-tindetheus: ## Download dependences.
	@cd ${TINDETHEUS_ROOT} && \
	[[ -d tinder ]] || mkdir tinder && \
	[[ -d pynder/pynder ]] || git submodule update --init --recursive && \
	if [[ ! -d .venv ]]; then \
		python3 -m venv .venv; \
		.venv/bin/pip install --upgrade -r requirements.txt; \
		.venv/bin/pip install PyQt5; \
	fi
