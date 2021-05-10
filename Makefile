all: conda-update pip-tools

# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

conda-update:
	conda env update --prune -f environment.yml

# Compile exact pip packages
pip-tools:
	pip install pip-tools
	pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html requirements/prod.in && pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt

# Example training command
# train:

# Lint
lint:
	tasks/lint.sh
