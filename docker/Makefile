BASE_NAME := 660440363484.dkr.ecr.us-west-2.amazonaws.com
REPO_NAME := opencap
PROD_BRANCH := main

# Determine the branch name
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# Determine image tag based on branch
ifeq ($(CURRENT_BRANCH),$(PROD_BRANCH))
    OPENCAP_IMAGE_TAG := opencap
	OPENPOSE_IMAGE_TAG := openpose
	MMPOSE_IMAGE_TAG := mmpose
else
    OPENCAP_IMAGE_TAG := opencap-dev
	OPENPOSE_IMAGE_TAG := openpose-dev
	MMPOSE_IMAGE_TAG := mmpose-dev
endif

# Get git commit hash info to pass it into container
GIT_COMMIT_HASH=$(shell git rev-parse --short HEAD)

.PHONY: build
build:
	wget -c -O ../mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
	wget -c -O ../mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
	
	docker build --build-arg GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) -t $(BASE_NAME)/$(REPO_NAME)/$(OPENCAP_IMAGE_TAG) .. -f Dockerfile
	docker build -t $(BASE_NAME)/$(REPO_NAME)/$(OPENPOSE_IMAGE_TAG) .. -f openpose/Dockerfile
	docker build -t $(BASE_NAME)/$(REPO_NAME)/$(MMPOSE_IMAGE_TAG) .. -f mmpose/Dockerfile

.PHONY: push
push:
	aws ecr get-login-password --region us-west-2 --profile opencap | docker login --username AWS --password-stdin 660440363484.dkr.ecr.us-west-2.amazonaws.com

	docker push $(BASE_NAME)/$(REPO_NAME)/$(OPENCAP_IMAGE_TAG)
	docker push $(BASE_NAME)/$(REPO_NAME)/$(OPENPOSE_IMAGE_TAG)
	docker push $(BASE_NAME)/$(REPO_NAME)/$(MMPOSE_IMAGE_TAG)

.PHONY: run
run:
	aws ecr get-login-password --region us-west-2 --profile opencap | docker login --username AWS --password-stdin 660440363484.dkr.ecr.us-west-2.amazonaws.com
	docker-compose up