//! Live audio to text with local models, as Plexus activations.
//!
//! A host service opens one [`Transcription`] and registers its activation
//! beside its own; its gate and its login decide who may call it, by the
//! roles the activations declare ([`ROLE`], `transcriber`; `admin` holds
//! every grant):
//!
//! ```no_run
//! use plexus_rpc::core::plexus::DynamicHub as DynamicService;
//! use plexus_transcription::{Options, Transcription, TranscriptionActivation};
//!
//! # fn main() -> anyhow::Result<()> {
//! let t = Transcription::open(Options::default_dirs()?)?;
//! let service = DynamicService::new("myapp").register(TranscriptionActivation::new(t));
//! // `myapp transcription listen …`, `myapp transcription models list`, …
//! # Ok(()) }
//! ```
//!
//! The library has no login and serves nothing: `server/` is the binary that
//! serves it on its own, behind plexus-login.

pub mod activation;
pub mod audio;
pub mod catalog;
pub mod config;
pub mod download;
pub mod engine;
pub mod listen;
pub mod store;

use std::sync::{Arc, Mutex};
use std::time::Instant;

pub use activation::{ModelsActivation, SpeechActivation, TranscriptionActivation, ROLE};
pub use catalog::Catalog;
pub use config::Config;
pub use listen::ListenEvent;
pub use store::Options;

/// One transcription: its models directory, its config, its loaded models and
/// its decode pool. Cheap to clone; every activation made from it shares them.
#[derive(Clone)]
pub struct Transcription {
    pub(crate) store: Arc<store::Store>,
    pub(crate) release: Arc<Mutex<Option<(Instant, Arc<Vec<activation::Asset>>)>>>,
}

impl Transcription {
    /// Opens the models directory (made if absent) and the kept config.
    pub fn open(opts: Options) -> anyhow::Result<Self> {
        Ok(Transcription { store: Arc::new(store::Store::open(opts)?), release: Arc::default() })
    }

    /// The `transcription` activation (`listen`, child `models`).
    pub fn activation(&self) -> TranscriptionActivation {
        TranscriptionActivation::new(self.clone())
    }

    /// `listen` alone, as `speech`.
    pub fn speech(&self) -> SpeechActivation {
        SpeechActivation::new(self.clone())
    }

    /// `models` alone.
    pub fn models(&self) -> ModelsActivation {
        ModelsActivation::new(self.clone())
    }

    /// The config `listen` runs under now.
    pub fn config(&self) -> Config {
        self.store.config()
    }

    /// Loads `model` now (with the config's threads), so the first `listen`
    /// does not wait for it.
    pub async fn warm(&self, model: &str) -> Result<(), String> {
        let e = self.store.recognizer(model)?.clone();
        let threads = self.store.config().resolve(model, e.kind).threads;
        self.store.engine(model, threads).await.map(|_| ())
    }
}
