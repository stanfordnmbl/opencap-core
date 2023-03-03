.PHONY: build
build:
	wget -c -O ../mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
	wget -c -O ../mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
	docker build -t opencap-dev .. -f Dockerfile
	docker build -t openpose-dev openpose --no-cache
	docker build -t mmpose-dev .. -f mmpose/Dockerfile

.PHONY: run
run:
	docker-compose up --remove-orphans
