.PHONY: build
build:
	wget -c -O ../mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
	wget -c -O ../mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
	docker build -t opencap-dev .. -f Dockerfile
	docker build -t openpose-dev .. -f openpose/Dockerfile
	docker build -t mmpose-dev .. -f mmpose/Dockerfile

# Push the image to ECR after building dev locally
.PHONY: push
push:
	aws ecr get-login-password --region us-west-2 --profile opencap | docker login --username AWS --password-stdin 660440363484.dkr.ecr.us-west-2.amazonaws.com
	docker push 660440363484.dkr.ecr.us-west-2.amazonaws.com/opencap/opencap-dev
	docker push 660440363484.dkr.ecr.us-west-2.amazonaws.com/opencap/openpose-dev
	docker push 660440363484.dkr.ecr.us-west-2.amazonaws.com/opencap/mmpose-dev

.PHONY: run
run:
	docker-compose up --remove-orphans
