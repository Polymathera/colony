

```shell
cd colony/test_runs

../python/colony/cli/polymath init-config --output my_analysis.yaml


colony-env down && \
colony-env up --workers 3 && \
colony-env run --local-repo /home/anassar/workspace/agents/crewAI --config my_analysis.yaml --verbose
```

