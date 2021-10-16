[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](meta/CODE_OF_CONDUCT.md)

# research-to-applied-ml

Curated papers, articles, and blogs on **data science & machine learning in production**, and then **implement**.

## License
See the [LICENSE](LICENSE.md) file for license rights and limitations.

## Contributing

Please check [CONTRIBUTING.md](meta/CONTRIBUTING.md).

## Directory

- `scripts/`: scripts for building/running
- `.env`: env file used in scripts
- `Makefile`: GNU Make Makefile as quick command entrypoint

## How to Use

### Prerequisite

- The environment for build needs to be linux/amd64 or macos/amd64
- The environemnt for build needs [docker engine installed](https://docs.docker.com/engine/install/)
- have [docker-compose](https://docs.docker.com/compose/install/) installed
- The environemnt for build needs GNU `make` > 3.8 installed
- The environemnt for build needs `bash` shell
- The environemnt for build needs `python ~= 3.8` installed

### Install dependency

We suggest create python environment using `virtualenv`:

```bash
virtualenv -p python venv
```

Then can install requirements with:

```bash
pip install requirements.txt
```

## TODO

## Ref

- https://github.com/eugeneyan/applied-ml
- https://github.com/eugeneyan/ml-surveys