# transcription — for an agent working here

README.md is the reference (methods, flags, the config file, the library's
API for a host, the tests). This file is what you need on top of it. The
house rules in `~/AGENTS.md` hold: say *service*, never *hub*; menger
tickets; never `git stash`.

## Commands

```sh
export CARGO_TARGET_DIR=~/.cache/cargo-target/TRANSCRIBE-2   # or your ticket's
cargo test -p transcription --lib
cargo test --release -p transcription-server --test listen    # one file at a time per change
cargo test --release -p transcription-server                  # the whole gate
cargo test --release -p transcription-server --test perf -- --ignored --nocapture
cargo run --release --example decode -- MODELS_DIR MODEL FILE.s16 [CHUNK] [--realtime]

# a server of your own: its own port and data dir (never 4450–4479: menger's)
PLEXUS_NO_REGISTRY=1 target/release/transcription --port 4481 --data-dir /tmp/tr-dev \
  --auth-provider key --admin "$(synapse _key show builder)"
SYNAPSE_NO_HOLD=1 synapse-4.0.0-input -P 4481 _login transcription --key builder --as builder
```

`synapse-4.0.0-input` (`~/.local/bin`) is synapse main with `--input`; the
default `synapse` link is older and has no `--input`. A loop over throwaway
servers sets `SYNAPSE_NO_HOLD=1`.

## Conventions

- **Two packages.** `transcription` (the root, `src/`) is the library: the
  activations, the engine, the store. It has no login and serves nothing —
  a host registers it (README "A host service"). `transcription-server`
  (`server/`) is the binary and the only place plexus-login and
  plexus-transport are used. Keep it that way.
- **The plexus crates are the 0.8 line by sibling path**, on `main`; never
  edit those checkouts from here. plexus-login's `main` pins plexus-core
  0.7 from crates.io, which a `[patch]` cannot replace with 0.8 (cargo
  leaves an incompatible version unused), so `overlay/plexus-login` is a
  manifest that compiles `../plexus-login/src` over the 0.8 siblings. Its
  dependencies mirror plexus-login's; when plexus-login moves to 0.8,
  delete the overlay and depend on `../plexus-login`.
- **The engine is behind `engine::Engine`/`Session`.** sherpa-onnx today
  (`engine/sherpa.rs`); a new engine (whisper.cpp, a GPU build —
  TRANSCRIBE-3) implements the two traits and is chosen in `engine::load`.
  A session's calls block: they run on the decode pool
  (`spawn_blocking` under the pool's semaphore), never on an async task.
- **Back-pressure is credit.** `listen` reads no input while its decode
  runs. Never buffer input unboundedly or drop it.
- **Every event enum** is `#[serde(tag = "type", rename_all = "snake_case")]`
  with an in-band `Error { message }`; refusals the method makes are in
  band, the gate's are the error frame.
- **Model ids** are `[a-z0-9-]` — no `.`: an id is a bare key in the
  config's `[models.<id>]`.
- **Models and audio of people are never committed.** Models live in the
  models directory (`~/.local/share/transcription/models`); test audio is
  synthetic (espeak-ng), its making written beside it.
- **Commits** say what and why, one per ticket, ending with the
  `Co-Authored-By` line; merge to `main` with `--no-ff`. Local only: no
  remote until the owner names one.
