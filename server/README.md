# run in Docker

## set environment variables

- OpenAI API key
- AWS creds:
```bash
eval "$(aws configure export-credentials --profile sandbox_dev --format env)"
```

## run

```bash
# from repo root
docker-compose up
```


## rebuild and run on code change

```bash
# from repo root
docker-compose up --build
```

## run client

```bash
cd ui/
yarn start
```
