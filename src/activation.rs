//! The activations: `listen`, and `models` (`list`, `search`, `download`,
//! `config`).
//!
//! Two shapes over one [`Transcription`]:
//!
//! - [`TranscriptionActivation`] (`transcription`): `listen`, with `models`
//!   as its child — what a host service registers beside its own
//!   activations: `<service> transcription listen`,
//!   `<service> transcription models list`.
//! - [`SpeechActivation`] (`speech`) and [`ModelsActivation`] (`models`) at a
//!   service's root — what this repository's own server registers on the
//!   service named `transcription`: `transcription speech listen`,
//!   `transcription models list`.
//!
//! Who may: [`ROLE`] (`transcriber`) holds `listen`, `models.list` and
//! `models.search`; `models.download` and `models.config` are declared for
//! no role, so only `admin` (which holds every grant) may.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use plexus_core::plexus::Input;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::audio::Format;
use crate::catalog::{matches, Family, Kind};
use crate::download::{download, Step};
use crate::listen::{listen, Ask, ListenEvent};

/// The role every activation here declares: who may listen and read the
/// models (`admin` may do everything).
pub const ROLE: &str = "transcriber";

// ── listen ───────────────────────────────────────────────────────────────

/// Live audio to text with local models; `models` manages them.
#[plexus_macros::activation(namespace = "transcription", version = "0.1.0", roles(transcriber = ["listen"]))]
impl TranscriptionActivation {
    /// Transcribe audio as it streams in: raw PCM in (binary frames, `--input -`), `partial` and `final` text out on the same call.
    #[plexus_macros::method(
        input_window = 64,
        params(
            model = "The model (`models list`); default: the config's default_model",
            rate = "Input sample rate, Hz (default 16000)",
            format = "Input sample format: s16le (default) or f32le",
            channels = "Input channels, interleaved (default 1); averaged to mono",
            audio = "The PCM itself"
        )
    )]
    pub async fn listen(&self, model: Option<String>, rate: Option<u32>, format: Option<Format>, channels: Option<u16>, audio: Input<Bytes>) -> impl Stream<Item = ListenEvent> + Send + 'static {
        listen(self.0.store.clone(), Ask { model, rate, format, channels }, audio)
    }

    /// The models: list, search, download, config.
    #[plexus_macros::child]
    fn models(&self) -> ModelsActivation {
        ModelsActivation(self.0.clone())
    }
}

/// `transcription`'s activation over a [`Transcription`].
#[derive(Clone)]
pub struct TranscriptionActivation(pub(crate) crate::Transcription);

impl TranscriptionActivation {
    pub fn new(t: crate::Transcription) -> Self {
        TranscriptionActivation(t)
    }
}

/// `listen` alone, for a service whose root holds `models` beside it.
#[derive(Clone)]
pub struct SpeechActivation(pub(crate) crate::Transcription);

impl SpeechActivation {
    pub fn new(t: crate::Transcription) -> Self {
        SpeechActivation(t)
    }
}

/// Live audio to text with local models.
#[plexus_macros::activation(namespace = "speech", version = "0.1.0", roles(transcriber = ["listen"]))]
impl SpeechActivation {
    /// Transcribe audio as it streams in: raw PCM in (binary frames, `--input -`), `partial` and `final` text out on the same call.
    #[plexus_macros::method(
        input_window = 64,
        params(
            model = "The model (`models list`); default: the config's default_model",
            rate = "Input sample rate, Hz (default 16000)",
            format = "Input sample format: s16le (default) or f32le",
            channels = "Input channels, interleaved (default 1); averaged to mono",
            audio = "The PCM itself"
        )
    )]
    pub async fn listen(&self, model: Option<String>, rate: Option<u32>, format: Option<Format>, channels: Option<u16>, audio: Input<Bytes>) -> impl Stream<Item = ListenEvent> + Send + 'static {
        listen(self.0.store.clone(), Ask { model, rate, format, channels }, audio)
    }
}

// ── models ───────────────────────────────────────────────────────────────

/// One model, as `models list` and `models search` say it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelInfo {
    pub id: String,
    /// The engine that runs it.
    pub engine: String,
    pub kind: Kind,
    pub family: Family,
    pub languages: Vec<String>,
    /// The archive's size, bytes.
    pub size: u64,
    pub installed: bool,
    /// Whether `listen` runs it with no `--model`.
    pub default: bool,
    /// Loaded in memory now.
    pub loaded: bool,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ListEvent {
    /// A model of the catalog.
    Model(ModelInfo),
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SearchEvent {
    /// A catalog model matching the query.
    Model(ModelInfo),
    /// An asset of the sherpa-onnx `asr-models` release matching the query.
    Asset {
        name: String,
        size: u64,
        url: String,
        /// The catalog id that downloads it, when there is one.
        catalog_id: Option<String>,
    },
    /// Something to know about the answer (the release could not be read).
    Note { message: String },
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DownloadEvent {
    /// Bytes of the archive on disk, of its size.
    Progress { id: String, bytes: u64, total: u64 },
    /// A cut download, picked up where it stopped.
    Resumed { id: String, from: u64 },
    /// The archive's size holds, and its sha256 when the catalog knows it.
    Verified { id: String, sha256: String, sha256_checked: bool },
    /// Extracted, checked, ready for `listen`.
    Installed { id: String, path: String },
    /// It was installed already; nothing was fetched.
    AlreadyInstalled { id: String, path: String },
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConfigEvent {
    /// The config now: its TOML, and where it is kept.
    Config { toml: String, path: Option<String>, applied: bool },
    Error { message: String },
}

macro_rules! client_error {
    ($($t:ty),*) => {$(
        impl plexus_client::FromClientError for $t {
            fn from_client_error(error: plexus_client::ClientError) -> Self {
                Self::Error { message: error.to_string() }
            }
        }
    )*};
}
client_error!(ListEvent, SearchEvent, DownloadEvent, ConfigEvent);

/// The models a service has: what is known, what is installed, fetching
/// one, and the config `listen` runs under.
#[derive(Clone)]
pub struct ModelsActivation(pub(crate) crate::Transcription);

impl ModelsActivation {
    pub fn new(t: crate::Transcription) -> Self {
        ModelsActivation(t)
    }

    fn info(&self, e: &crate::catalog::Entry) -> ModelInfo {
        let store = &self.0.store;
        let default = store.config().default_model;
        ModelInfo {
            id: e.id.clone(),
            engine: "sherpa-onnx".into(),
            kind: e.kind,
            family: e.family,
            languages: e.languages.clone(),
            size: e.size,
            installed: store.installed(&e.id).is_some(),
            default: e.id == default,
            loaded: store.loaded_ids().contains(&e.id),
            description: e.description.clone(),
        }
    }
}

/// Local speech models: list, search, download and configure them.
#[plexus_macros::activation(namespace = "models", version = "0.1.0", roles(transcriber = ["list", "search"]))]
impl ModelsActivation {
    /// Every model the catalog knows, installed or not, and which is the default.
    #[plexus_macros::method]
    pub async fn list(&self) -> impl Stream<Item = ListEvent> + Send + 'static {
        let items: Vec<ListEvent> = self.0.store.catalog().entries().iter().map(|e| ListEvent::Model(self.info(e))).collect();
        futures::stream::iter(items)
    }

    /// Catalog models and sherpa-onnx `asr-models` release assets whose names match the query.
    #[plexus_macros::method(params(query = "Words to match, e.g. `parakeet` or `zipformer en`"))]
    pub async fn search(&self, query: String) -> impl Stream<Item = SearchEvent> + Send + 'static {
        let this = self.clone();
        async_stream::stream! {
            let q = query.trim().to_string();
            if q.is_empty() {
                yield SearchEvent::Error { message: "the query is empty".into() };
                return;
            }
            let store = this.0.store.clone();
            for e in store.catalog().entries() {
                if matches(&q, &format!("{} {} {}", e.id, e.archive, e.description)) {
                    yield SearchEvent::Model(this.info(e));
                }
            }
            match this.0.release_assets().await {
                Ok(assets) => {
                    for (name, size, url) in assets.iter().filter(|a| matches(&q, &a.0)) {
                        let catalog_id = store.catalog().entries().iter().find(|e| &e.archive == name).map(|e| e.id.clone());
                        yield SearchEvent::Asset { name: name.clone(), size: *size, url: url.clone(), catalog_id };
                    }
                }
                Err(message) => yield SearchEvent::Note { message },
            }
        }
    }

    /// Download a model (and the voice detector, the first time): progress streams, then it is verified, extracted and installed. A cut download resumes.
    #[plexus_macros::method(params(id = "The model id (`models list`)"))]
    pub async fn download(&self, id: String) -> impl Stream<Item = DownloadEvent> + Send + 'static {
        let store = self.0.store.clone();
        async_stream::stream! {
            let entry = match store.catalog().get(&id) {
                Some(e) => e.clone(),
                None => {
                    yield DownloadEvent::Error { message: format!("no model `{id}` (the models: {})", store.catalog().ids()) };
                    return;
                }
            };
            let mut wanted = vec![entry.clone()];
            if entry.kind != Kind::Vad {
                if let Some(vad) = store.catalog().get(crate::catalog::VAD_ID) {
                    if store.installed(&vad.id).is_none() {
                        wanted.insert(0, vad.clone());
                    }
                }
            }
            for e in wanted {
                let mut steps = Box::pin(download(store.clone(), e.clone()));
                while let Some(step) = steps.next().await {
                    let id = e.id.clone();
                    match step {
                        Ok(Step::Progress { bytes, total }) => yield DownloadEvent::Progress { id, bytes, total },
                        Ok(Step::Resumed { from }) => yield DownloadEvent::Resumed { id, from },
                        Ok(Step::Verified { sha256, checked }) => yield DownloadEvent::Verified { id, sha256, sha256_checked: checked },
                        Ok(Step::Installed { path }) => yield DownloadEvent::Installed { id, path: path.display().to_string() },
                        Ok(Step::AlreadyInstalled { path }) => yield DownloadEvent::AlreadyInstalled { id, path: path.display().to_string() },
                        Err(message) => {
                            yield DownloadEvent::Error { message: format!("{id}: {message}") };
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Set the config from a TOML file's contents (`--config @transcription.toml`), checked and kept, for the next listen; with no `--config`, show it.
    #[plexus_macros::method(params(config = "The TOML text; absent: answer the current config"))]
    pub async fn config(&self, config: Option<String>) -> ConfigEvent {
        let store = &self.0.store;
        let path = store.opts.config_path.as_ref().map(|p| p.display().to_string());
        match config {
            None => ConfigEvent::Config { toml: store.config().to_toml(), path, applied: false },
            Some(text) => match store.set_config(&text) {
                Ok(c) => ConfigEvent::Config { toml: c.to_toml(), path, applied: true },
                Err(message) => ConfigEvent::Error { message },
            },
        }
    }
}

// ── the release's assets, for search ─────────────────────────────────────

/// How long the release's asset list is kept before it is read again.
const RELEASE_TTL: Duration = Duration::from_secs(600);

pub(crate) type Asset = (String, u64, String);

impl crate::Transcription {
    async fn release_assets(&self) -> Result<Arc<Vec<Asset>>, String> {
        let Some(api) = self.store.opts.release_api.clone() else {
            return Err("the release is not searched here (no release API configured)".into());
        };
        if let Some((at, list)) = self.release.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            if at.elapsed() < RELEASE_TTL {
                return Ok(list);
            }
        }
        #[derive(Deserialize)]
        struct Release {
            assets: Vec<A>,
        }
        #[derive(Deserialize)]
        struct A {
            name: String,
            size: u64,
            browser_download_url: String,
        }
        let resp = self
            .store
            .http
            .get(&api)
            .header("accept", "application/vnd.github+json")
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(|e| format!("the release could not be read ({e}): the catalog's matches only"))?;
        if !resp.status().is_success() {
            return Err(format!("the release could not be read (HTTP {}): the catalog's matches only", resp.status()));
        }
        let r: Release = resp.json().await.map_err(|e| format!("the release could not be read ({e}): the catalog's matches only"))?;
        let list: Arc<Vec<Asset>> = Arc::new(r.assets.into_iter().map(|a| (a.name, a.size, a.browser_download_url)).collect());
        *self.release.lock().unwrap_or_else(|p| p.into_inner()) = Some((Instant::now(), list.clone()));
        Ok(list)
    }
}
