FROM golang:1.26-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app

COPY pkg/jwt /app/pkg/jwt
COPY pkg/pb/text_storage /app/pkg/pb/text_storage
COPY services/job-processor/go.mod services/job-processor/go.sum ./services/job-processor/

WORKDIR /app/services/job-processor
RUN go mod download

COPY services/job-processor ./

RUN CGO_ENABLED=0 GOOS=linux go build -o /bin/job-processor ./cmd

FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /bin/job-processor /bin/job-processor

EXPOSE 8082

ENTRYPOINT ["/bin/job-processor"]
