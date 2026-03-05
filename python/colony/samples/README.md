

```shell
cd colony/test_runs

../python/colony/cli/polymath init-config --output my_analysis.yaml

source ../python/colony/cli/deploy/.env

colony-env down && \
colony-env up --workers 3 && \
colony-env run --local-repo /home/anassar/workspace/agents/crewAI --config my_analysis.yaml --verbose

colony-env down && \
colony-env up --workers 3 && \
colony-env run --local-repo /home/anassar/workspace//distributed/ray --config my_analysis.yaml --verbose
```

