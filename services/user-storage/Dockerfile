FROM golang:1.26-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app

COPY pkg/jwt /app/pkg/jwt
COPY services/user-storage/go.mod services/user-storage/go.sum ./services/user-storage/

WORKDIR /app/services/user-storage
RUN go mod download

COPY services/user-storage ./

RUN CGO_ENABLED=0 GOOS=linux go build -o /bin/user-storage ./cmd

FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /bin/user-storage /bin/user-storage

EXPOSE 8081

ENTRYPOINT ["/bin/user-storage"]
