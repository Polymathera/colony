
## Efficient data sharing

`PolymatheraObjectStore` for sharing immutable (read-only) large data (e.g., context pages) between components without serialization/deserialization overhead through all the deployment layers (proxy actors, routers, etc.). Only the metadata (e.g., object IDs) is passed around.
