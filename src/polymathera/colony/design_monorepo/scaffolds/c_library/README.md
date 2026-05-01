# ${name}

${description}

C library scaffolded into ``tools/${purpose}/${name}/`` of a Polymathera
Colony design monorepo. Replace the smoke-test surface in
``include/${name_snake}.h`` and ``src/${name_snake}.c`` with the tool's
real implementation; keep the CMake / CTest harness so the tool-building
pool can run regression and benchmark suites uniformly.

## Build & test

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Provenance

Scaffolded ${iso_date} by ${author}. Licence: ${license}.
