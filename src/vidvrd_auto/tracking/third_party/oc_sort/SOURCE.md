# OC-SORT source

- Repository: https://github.com/noahcao/OC_SORT
- Commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- Files: `ocsort.py`, `association.py`, `kalmanfilter.py`
- License: MIT, copied as `LICENSE`

Normalized LF SHA-256:

- `ocsort.py`: `c900be251d1ce01483880ad5b144195072153cb0810d2f2d0c5db256628982db`
- `association.py`: `71c5e97b6f93472b98f80cb754294daa7606d78fe40aa8a838a938e51507d504`
- `kalmanfilter.py`: `96c859ec913640e3e6ebb1e5cdc2c6f0b94bdb140a6d54c3a8586868e5dc374a`

The upstream Python files are vendored without semantic edits. Project-specific
class handling, metadata, clipping, and output conversion live outside this
directory in `tracking/ocsort/adapter.py`.
