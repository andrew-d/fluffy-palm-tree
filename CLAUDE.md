# Notes for Claude

## Running tests

Always run the Go test suite with `-p=1` so packages execute one at a time:

```
go test -p=1 ./...
```

Several test packages load the full ~1.5B-parameter model (`privacyfilter.LoadModel("./model")`). The default `go test` parallelism (`GOMAXPROCS`) fans those out across packages, and loading the weights multiple times concurrently will OOM the machine. `-p=1` serializes package execution so the model is held by at most one test binary at a time.
