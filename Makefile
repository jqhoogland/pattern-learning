PROJECT_ID ?= research
DOCKER_USER ?= jqhoogland
DOCKER_REPO ?= $(DOCKER_USER)/$(PROJECT_ID)
SWEEP_IMAGE ?= dominoes-sweep

.PHONY: build sweepid

sweepid: 
	python dominoes/sweep_create.py

build:
	docker build --build-arg WANDB_SWEEP_ID=$(WANDB_SWEEP_ID) -t $(PROJECT_ID) .

tag:
	docker tag $(PROJECT_ID):latest $(DOCKER_REPO):latest

push: 
	docker push $(DOCKER_REPO):latest

sweep:
	kubectl create job --from=cronjob/$(SWEEP_IMAGE) $(SWEEP_IMAGE)-$(shell date +%s)