# List of targets that we want to forward to the docs/Makefile
DOCS_TARGETS = html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest gettext

# Rule for forwarding the targets to the docs/Makefile
$(DOCS_TARGETS):
	$(MAKE) -C docs $@


fixpath:
	export PYTHONPATH=$(shell pwd):$$PYTHONPATH; \
	echo $$PYTHONPATH


# Default rule (optional): By default, let's build the HTML version of docs.
# You can change 'html' to any other target you prefer as default.
default: html

.PHONY: $(DOCS_TARGETS) default
