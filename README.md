# transcription

Live audio to text with local models, as a Plexus RPC service (TRANSCRIBE-2).
A signed-in caller holding the `transcriber` role pipes raw PCM into
`speech listen` and reads `partial` and `final` text back **on the same call**
while it speaks; `models list | search | download | config` manage the models.
Every model runs on this machine (CPU, sherpa-onnx); there are no cloud calls.

```
pw-record ─PCM─▶ synapse --input - ═binary frames═▶ speech.listen ─▶ 16 kHz mono ─▶ Silero VAD + recognizer
                                    ◀══ partial … final {text, start_ms, end_ms} … done
```

## Layout

| path | what |
| -- | -- |
| `src/` | **the library** (`transcription`): the activations, the engine, the models store — no login, serves nothing |
| `src/activation.rs` | `TranscriptionActivation` (`listen` + child `models`), `SpeechActivation` (`listen`), `ModelsActivation`; `ROLE` |
| `src/listen.rs` | the pipeline: input → convert → decode pool → events |
| `src/engine/` | the `Engine` / `Session` traits; `sherpa.rs`, sherpa-onnx + Silero VAD |
| `src/audio.rs` | s16le/f32le, any rate and channels → 16 kHz mono f32 (rubato) |
| `src/catalog.rs` | the models known: ids, archives, sizes, sha256 |
| `src/config.rs` | `transcription.toml` |
| `src/store.rs`, `src/download.rs` | the models directory, loaded models, the pool; download/verify/extract |
| `server/` | **the binary** (`transcription`): the library on a service of its own behind plexus-login |
| `overlay/plexus-login/` | plexus-login's sources compiled over the 0.8 siblings (manifest only; AGENTS.md says why) |
| `examples/decode.rs` | a file through the engine directly, with its RTF |
| `transcription.toml` | the example config |

## Build

The plexus crates are the unpublished 0.8 line, by sibling path
(`../plexus-core`, `../plexus-macros`, `../plexus-transport`,
`../plexus-auth-core`, `../plexus-login`), on their `main`.

```sh
cargo build --release -p transcription-server     # target/release/transcription
```

sherpa-onnx's crate downloads its prebuilt static libraries (onnxruntime
inside) at build time: no cmake, no system packages.

## Run and sign in

```sh
transcription --port 4480 --auth-provider key --admin "$(synapse _key show builder)"
synapse -P 4480 _login transcription --key builder --as builder
```

Login is always on. `--auth-provider key` admits agent keys (synapse's
`_login`); `idp=…`, `oidc:<name>=…`, `auth0=…`, `workos=…`, `nostr=…` are
plexus-login's other providers. Roles are plexus-login's `SimpleAccess`:
global, in memory. The `--admin` principals are admin from the start;
anyone else who signs in waits (`auth waiting`) until an admin grants them
a role; a restart forgets every grant but the admins'.

```sh
synapse -P 4480 transcription auth waiting
synapse -P 4480 transcription auth grant --principal key:… --role transcriber --scope '*'
```

| role | may |
| -- | -- |
| `transcriber` | `speech.listen`, `models.list`, `models.search` |
| `admin` | everything: also `models.download`, `models.config`, `auth.*` |

A caller with neither is refused by the gate, typed (`forbidden`).

### Every flag

| flag | env | default |
| -- | -- | -- |
| `--port` | `TRANSCRIPTION_PORT` | 4480 |
| `--bind` | `TRANSCRIPTION_BIND` | 127.0.0.1 |
| `--data-dir` | `TRANSCRIPTION_DATA_DIR` | `$XDG_DATA_HOME/transcription`, else `~/.local/share/transcription` |
| `--models-dir` | `TRANSCRIPTION_MODELS_DIR` | `<data-dir>/models` |
| `--catalog FILE` | `TRANSCRIPTION_CATALOG` | more `[[model]]` entries (the fields of `catalog::Entry`), or replacements by id |
| `--mirror URL` | `TRANSCRIPTION_MIRROR` | fetch the built-in archives from URL/`<archive>` instead of GitHub |
| `--offline-search` | | `models search` reads the catalog only |
| `--warm ID` | | load a model at start |
| `--auth-provider SPEC` | `TRANSCRIPTION_AUTH_PROVIDERS` (`;`) | required |
| `--admin PRINCIPAL` | `TRANSCRIPTION_ADMINS` (`;`) | |
| `--session-hours` | `TRANSCRIPTION_SESSION_HOURS` | 12 |

`PLEXUS_NO_REGISTRY=1` keeps it out of the local registry.

## The walk

```bash
transcription --port 4480 --auth-provider key --admin "$(synapse _key show builder)"
synapse -P 4480 _login transcription --key builder --as builder
synapse -P 4480 transcription models search --query parakeet
synapse -P 4480 transcription models download --id nemotron-en-560ms      # progress streams
synapse -P 4480 transcription models config --config @transcription.toml
pw-record --format s16 --rate 16000 --channels 1 - \
  | synapse -P 4480 transcription speech listen --model nemotron-en-560ms --input - --chunk 640
# {"type":"partial","text":"So the plan for"}
# {"type":"final","text":"So the plan for today is the kiln","start_ms":74,"end_ms":3664}
```

`synapse` here is one with `--input` (4.0.0 and later; on this machine
`~/.local/bin/synapse-4.0.0-input`). `--input -` streams stdin as binary
frames of at most `--chunk` bytes, under the service's credit; at end of
input the last utterance is flushed as `final`, then `done`. `-j` prints the
raw items for a script.

From a file: `ffmpeg -i talk.wav -ar 16000 -ac 1 -f s16le - | synapse … speech listen --input -`,
or say the file's shape: `… speech listen --rate 48000 --channels 2 --format s16le --input -`.

**`speech listen`, not `listen`.** A Plexus service routes
`<activation>.<method>`; its own root answers only its built-in methods
(`schema`, `hash`, …), so a method cannot sit at `transcription listen` today.
`models` is an activation, so `transcription models …` reads as asked.

## Methods

### `speech listen` (transcriber)

| param | |
| -- | -- |
| `--model ID` | default: the config's `default_model` |
| `--rate HZ` | input sample rate, default 16000; any other is resampled |
| `--format s16le\|f32le` | default s16le |
| `--channels N` | default 1; interleaved channels are averaged |
| `--input -` | the PCM (an `Input<Bytes>`: binary frames) |

| event | when |
| -- | -- |
| `partial {text}` | the open utterance so far, while it is spoken; the next partial or the final replaces it |
| `final {text, start_ms, end_ms}` | an utterance, done (Silero VAD segments it); ms from the call's first sample |
| `error {message}` | refused in band: no such model, not downloaded (naming `models download --id M`), a bad shape |

- **A streaming model** (Nemotron, zipformer): partials from the recognizer
  as audio arrives. One stream per listener runs the whole call (the model
  needs its left context). When the VAD ends a segment — after
  `vad.min_silence_ms` of silence, so nothing of the next utterance has
  reached the stream — the stream is given `final_tail_ms` of silence for
  its look-ahead and every token since the last final is the final.
- **An offline model** (Parakeet, Moonshine, Whisper): each VAD segment is
  decoded whole for its final; while speech is open the utterance so far is
  re-decoded every `partial_ms` of new audio for a partial.
- **Many listeners**: one loaded model, shared. Decodes run on a bounded
  pool (`decode_workers`) off the socket's tasks; a listener reads no more
  input while its decode runs, so when decode falls behind the service
  grants less credit and the client holds its audio — nothing is dropped.
- **A dropped socket** ends the call; nothing is kept.

### `models list` (transcriber)

Every catalog model: `id, engine, kind (streaming|offline|vad), family,
languages, size, installed, default, loaded, description`.

### `models search --query Q` (transcriber)

Catalog models matching every word of Q, then the assets of the sherpa-onnx
GitHub release `asr-models` matching Q (`asset {name, size, url,
catalog_id}`; `catalog_id` is the id that downloads it when the catalog has
one). Release unreachable: a `note`, and the catalog's matches.

### `models download --id M` (admin)

Streams `progress {id, bytes, total}`, `resumed {id, from}` when a cut
download picks up (HTTP `Range`), `verified {id, sha256, sha256_checked}`
(size always; sha256 when the catalog knows it), then `installed {id,
path}`. The first model also fetches `silero-vad`. Installed already:
`already_installed`, nothing fetched. A cut download keeps its bytes
(`<models>/.partial/`) and the next call resumes; an archive that does not
check is removed.

### `models config [--config @file.toml]` (admin)

With `--config`: the file's contents are checked — an unknown key refused
naming it, a value out of range refused naming it, the old config kept —
then kept (`<data-dir>/transcription.toml`) and applied to the **next**
`listen` (never one under way). With nothing: the current config.
Answers `config {toml, path, applied}` or `error {message}`.

## The config file

```toml
# transcription.toml — every key optional; an unknown one is refused, named.
default_model = "nemotron-en-560ms"   # listen with no --model
threads = 2                           # onnxruntime threads per decode
decode_workers = 8                    # decodes at once, all listeners (default: cores / threads)
partial_ms = 700                      # offline models: re-decode for a partial every this much new audio; 0: finals only

[vad]
threshold = 0.5         # speech probability, 0..1
min_silence_ms = 400    # silence that ends an utterance (and the final's latency)
min_speech_ms = 150     # shorter is not an utterance
max_speech_s = 20       # longer is cut

[models.parakeet-tdt-v2]       # per-model overrides: threads, partial_ms, final_tail_ms
partial_ms = 800

[models.nemotron-en-560ms]
threads = 2
final_tail_ms = 1200    # silence given a streaming model at an utterance's end, for its look-ahead
```

## Models

| id | kind | languages | archive size | notes |
| -- | -- | -- | -- | -- |
| `nemotron-en-560ms` | streaming | en | 464 MB | NVIDIA Nemotron Speech Streaming 0.6B int8, manifest's engine; the default |
| `nemotron-en-160ms`, `nemotron-en-80ms` | streaming | en | 464 MB | the same at shorter chunks |
| `zipformer-en-20m` | streaming | en | 128 MB | small and fast; poor on synthetic speech |
| `parakeet-tdt-v2` | offline | en | 482 MB | NVIDIA Parakeet TDT 0.6B v2 int8, punctuated |
| `parakeet-tdt-v3` | offline | 25 European | 487 MB | v3 |
| `moonshine-tiny-en`, `moonshine-base-en` | offline | en | 108 / 251 MB | Useful Sensors Moonshine int8 |
| `whisper-tiny-en` | offline | en | 118 MB | OpenAI Whisper tiny.en |
| `silero-vad` | vad | | 0.6 MB | fetched with the first model |

All are assets of the sherpa-onnx release `asr-models`
(<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>), each
under its own license (see each archive's README/LICENSE). The sha256s come
from GitHub's asset digests where it has one, else were measured from a
download here (`sha256_source` in the catalog). Models are kept in the
models directory and never committed.

## A host service: the library

The activations are a library a service registers beside its own, behind
its own gate and login (TRANSCRIBE-2's addition for plexus-ui):

```rust
use plexus_core::plexus::DynamicHub as DynamicService;
use transcription::{Options, Transcription, TranscriptionActivation, ROLE};

let t = Transcription::open(Options::default_dirs()?)?;   // or Options { models_dir, config_path, catalog, mirror, release_api }
let service = DynamicService::new("myapp")
    .with_auth(auth_setup)                                 // the host's login
    .register(TranscriptionActivation::new(t.clone()));    // = t.activation()
access.seed_roles(&service);                               // seeds `transcriber`
// myapp transcription listen …          (transcription.listen)
// myapp transcription models list       (transcription.models.list)
```

- `Transcription::open(Options) -> anyhow::Result<Transcription>` — the
  models directory, the config, the loaded models and the decode pool;
  cheap to clone, shared by every activation made from it.
  `Options::default_dirs()` is `~/.local/share/transcription/{models,transcription.toml}`,
  the built-in catalog and GitHub.
- `TranscriptionActivation::new(t)` (namespace `transcription`): `listen`
  with `models` as its static child. Or `SpeechActivation::new(t)`
  (`speech`) and `ModelsActivation::new(t)` (`models`) at the root, as the
  server does.
- Roles declared: `transcriber` on `listen`, `models.list`, `models.search`
  (`transcription::ROLE`); `models.download` and `models.config` are
  declared for no role, so only `admin` holds them. Under a gate, the
  child's are qualified by path (`transcription.models.list`).
- `t.warm(id).await` loads a model ahead of the first call; `t.config()`
  reads the config.
- Events: `transcription::ListenEvent` (`partial`, `final`, `error`) and
  `activation::{ListEvent, SearchEvent, DownloadEvent, ConfigEvent}`.

## Tests

```sh
cargo test -p transcription --lib                  # catalog, config, audio
cargo test --release -p transcription-server       # over a real socket (needs models, below)
cargo test --release -p transcription-server --test perf -- --ignored --nocapture   # the perf table
cargo run --release --example decode -- ~/.local/share/transcription/models nemotron-en-560ms server/tests/fixtures/walk.s16
```

The socket tests use the installed models in the default models directory
(or `TRANSCRIPTION_TEST_MODELS`): `nemotron-en-560ms`, `parakeet-tdt-v2`,
`silero-vad`; a missing one fails naming the download. The test clip is
synthetic speech (espeak-ng), `server/tests/fixtures/README.md`.

| test | asserts |
| -- | -- |
| `listen` | the clip's finals within WER 0.15 (Nemotron) and 0.10 (Parakeet), three VAD-segmented finals with times, partials; a partial before the input ends; 48 kHz stereo and f32le converted; a model not downloaded refused naming the download |
| `access` | no role → `forbidden` on every method; `transcriber` → listen, list, search served, download and config forbidden; admin served; the roles declared |
| `config` | an unknown key (top, `[vad]`, `[models.X]`), an unknown model, a value out of range — each refused naming it, the old config kept on disk and live; a good one kept, applied to the next listen, read back at start |
| `download` | against a local HTTP fixture: VAD first, the model cut halfway → error saying resume; the next call resumes with `Range`, verifies sha256, extracts, installs; a third says installed; a bad sha256 refused and the partial removed |
| `concurrency` | 8 listeners at once, each model: all finish, all within the WER bound, RSS growth bounded |
