//! The models directory, the config, the loaded models and the decode pool —
//! what every activation of one [`crate::Transcription`] shares.
//!
//! Layout under the models directory:
//!
//! ```text
//! <models>/<id>/…              an installed model: the archive's directory
//! <models>/<id>/.installed     written last, after the files are checked
//! <models>/silero-vad/silero_vad.onnx
//! <models>/.partial/<archive>  a download under way or cut (resumed)
//! <models>/.staging-<id>/      an extraction under way
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use tokio::sync::Semaphore;

use crate::catalog::{Catalog, Entry, Kind, VAD_ID};
use crate::config::Config;
use crate::engine::{self, Engine, Load};

/// What a [`crate::Transcription`] is opened with.
#[derive(Debug, Clone)]
pub struct Options {
    /// Where models are installed.
    pub models_dir: PathBuf,
    /// Where `models config` keeps the config (read at open when it is
    /// there). `None`: in memory only.
    pub config_path: Option<PathBuf>,
    /// The catalog (built-ins, plus any extra entries).
    pub catalog: Catalog,
    /// A base URL to fetch the built-in archives from instead of GitHub's.
    pub mirror: Option<String>,
    /// The release description `models search` reads (GitHub's API); `None`:
    /// search the catalog only.
    pub release_api: Option<String>,
}

impl Options {
    /// The defaults: `~/.local/share/transcription/models`, the config beside
    /// it, the built-in catalog, GitHub.
    pub fn default_dirs() -> anyhow::Result<Self> {
        let data = default_data_dir()?;
        Ok(Options {
            models_dir: data.join("models"),
            config_path: Some(data.join("transcription.toml")),
            catalog: Catalog::new(None)?,
            mirror: None,
            release_api: Some(crate::catalog::RELEASE_API.into()),
        })
    }
}

/// `$XDG_DATA_HOME/transcription`, else `~/.local/share/transcription`.
pub fn default_data_dir() -> anyhow::Result<PathBuf> {
    if let Some(x) = std::env::var_os("XDG_DATA_HOME").filter(|x| !x.is_empty()) {
        return Ok(PathBuf::from(x).join("transcription"));
    }
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("no HOME: pass --data-dir"))?;
    Ok(PathBuf::from(home).join(".local/share/transcription"))
}

pub struct Store {
    pub opts: Options,
    config: RwLock<Config>,
    loaded: Mutex<HashMap<String, (u32, Arc<dyn Engine>)>>,
    load_lock: tokio::sync::Mutex<()>,
    pool: RwLock<(u32, Arc<Semaphore>)>,
    pub(crate) downloading: Mutex<HashSet<String>>,
    pub(crate) http: reqwest::Client,
}

impl Store {
    pub fn open(opts: Options) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&opts.models_dir).map_err(|e| anyhow::anyhow!("{}: {e}", opts.models_dir.display()))?;
        let config = match &opts.config_path {
            Some(p) if p.is_file() => {
                let text = std::fs::read_to_string(p)?;
                Config::parse(&text, &opts.catalog).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?
            }
            _ => Config::default(),
        };
        let workers = config.decode_workers;
        let http = reqwest::Client::builder()
            .user_agent(concat!("transcription/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(20))
            .build()?;
        Ok(Store {
            opts,
            config: RwLock::new(config),
            loaded: Mutex::default(),
            load_lock: tokio::sync::Mutex::new(()),
            pool: RwLock::new((workers, Arc::new(Semaphore::new(workers as usize)))),
            downloading: Mutex::default(),
            http,
        })
    }

    pub fn catalog(&self) -> &Catalog {
        &self.opts.catalog
    }

    pub fn config(&self) -> Config {
        self.config.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Checks `text`, keeps it (the file first, so a refusal to write keeps
    /// the old one) and applies it to the next `listen`.
    pub fn set_config(&self, text: &str) -> Result<Config, String> {
        let c = Config::parse(text, self.catalog())?;
        if let Some(p) = &self.opts.config_path {
            write_atomic(p, c.to_toml().as_bytes()).map_err(|e| format!("{}: {e} (the old config is kept)", p.display()))?;
        }
        *self.config.write().unwrap_or_else(|p| p.into_inner()) = c.clone();
        let mut pool = self.pool.write().unwrap_or_else(|p| p.into_inner());
        if pool.0 != c.decode_workers {
            // Decodes under way finish on the old pool; new ones take this.
            *pool = (c.decode_workers, Arc::new(Semaphore::new(c.decode_workers as usize)));
        }
        Ok(c)
    }

    /// The decode pool: a permit per decode running.
    pub fn pool(&self) -> Arc<Semaphore> {
        self.pool.read().unwrap_or_else(|p| p.into_inner()).1.clone()
    }

    pub fn dir_of(&self, id: &str) -> PathBuf {
        self.opts.models_dir.join(id)
    }

    pub fn vad_file(&self) -> PathBuf {
        self.dir_of(VAD_ID).join("silero_vad.onnx")
    }

    /// The installed directory of `id`, if it is installed.
    pub fn installed(&self, id: &str) -> Option<PathBuf> {
        let dir = self.dir_of(id);
        dir.join(".installed").is_file().then_some(dir)
    }

    pub fn partial_file(&self, e: &Entry) -> PathBuf {
        self.opts.models_dir.join(".partial").join(&e.archive)
    }

    /// The model `id`, loaded with `threads` — once, shared by every
    /// listener; a load runs off the async threads.
    pub async fn engine(&self, id: &str, threads: u32) -> Result<Arc<dyn Engine>, String> {
        if let Some(e) = self.cached(id, threads) {
            return Ok(e);
        }
        let _one = self.load_lock.lock().await;
        if let Some(e) = self.cached(id, threads) {
            return Ok(e);
        }
        let entry = self.catalog().get(id).cloned().ok_or_else(|| format!("no model `{id}`"))?;
        let dir = self.installed(id).ok_or_else(|| not_downloaded(id))?;
        let load = Load { entry, dir, vad: self.vad_file(), threads };
        let started = std::time::Instant::now();
        let engine = tokio::task::spawn_blocking(move || engine::load(&load))
            .await
            .map_err(|e| format!("loading {id}: {e}"))?
            .map_err(|e| format!("loading {id}: {e:#}"))?;
        tracing::info!(model = id, threads, ms = started.elapsed().as_millis() as u64, "model loaded");
        self.loaded.lock().unwrap_or_else(|p| p.into_inner()).insert(id.to_string(), (threads, engine.clone()));
        Ok(engine)
    }

    fn cached(&self, id: &str, threads: u32) -> Option<Arc<dyn Engine>> {
        let loaded = self.loaded.lock().unwrap_or_else(|p| p.into_inner());
        loaded.get(id).filter(|(t, _)| *t == threads).map(|(_, e)| e.clone())
    }

    /// Ids loaded now.
    pub fn loaded_ids(&self) -> Vec<String> {
        self.loaded.lock().unwrap_or_else(|p| p.into_inner()).keys().cloned().collect()
    }

    /// Whether `id` is a recognizer the catalog holds.
    pub fn recognizer(&self, id: &str) -> Result<&Entry, String> {
        match self.catalog().get(id) {
            Some(e) if e.kind != Kind::Vad => Ok(e),
            _ => Err(format!("no model `{id}` (the models: {})", self.catalog().ids())),
        }
    }
}

pub fn not_downloaded(id: &str) -> String {
    format!("model `{id}` is not downloaded: run `models download --id {id}`")
}

/// Writes `bytes` to `path` whole: a temp file beside it, fsynced, renamed.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}
