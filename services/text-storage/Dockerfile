FROM golang:1.26-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app

COPY pkg/pb/text_storage /app/pkg/pb/text_storage
COPY services/text-storage/go.mod services/text-storage/go.sum ./services/text-storage/

WORKDIR /app/services/text-storage
RUN go mod download

COPY services/text-storage ./

RUN CGO_ENABLED=0 GOOS=linux go build -o /bin/text-storage ./cmd

FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /bin/text-storage /bin/text-storage

EXPOSE 50051

ENTRYPOINT ["/bin/text-storage"]
