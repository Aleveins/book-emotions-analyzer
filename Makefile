.PHONY: proto build up down logs clean

proto:
	buf generate

build:
	docker compose build

up:
	docker compose up -d

up-build:
	docker compose up -d --build

down:
	docker compose down

down-volumes:
	docker compose down -v

logs:
	docker compose logs -f

logs-%:
	docker compose logs -f $*

restart-%:
	docker compose restart $*

ps:
	docker compose ps
