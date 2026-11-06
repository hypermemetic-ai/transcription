//! A server in the test's process over a real socket, under the same login
//! the binary runs (`key`), with callers signed in by minting their session
//! on its table — the token synapse would hold after `_login`.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use plexus_rpc::client::{InputItem, MethodRef, StreamItem, Transport, WsTransport};
use plexus_rpc::core::plexus::AuthContext;
use serde_json::{json, Value};
use transcription::{Catalog, Options, Transcription};
use transcription_server::{start, Login, LoginArgs, Serving};

pub const ADMIN: &str = "key:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
pub const ALICE: &str = "key:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
pub const BOB: &str = "key:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

/// The installed models the tests run: `TRANSCRIPTION_TEST_MODELS`, else the
/// default models directory. A test that needs one says which is missing
/// and how to get it.
pub fn models_dir() -> PathBuf {
    std::env::var_os("TRANSCRIPTION_TEST_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| transcription::store::default_data_dir().unwrap().join("models"))
}

pub fn need(ids: &[&str]) {
    let dir = models_dir();
    for id in ids.iter().chain(&["silero-vad"]) {
        assert!(
            dir.join(id).join(".installed").is_file(),
            "model `{id}` is not installed in {} — `synapse … transcription models download --id {id}`, or set TRANSCRIPTION_TEST_MODELS",
            dir.display()
        );
    }
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

pub fn walk() -> Vec<u8> {
    std::fs::read(fixture("walk.s16")).unwrap()
}

pub fn walk_text() -> String {
    std::fs::read_to_string(fixture("walk.txt")).unwrap()
}

pub struct Server {
    pub login: Login,
    pub serving: Option<Serving>,
    pub addr: SocketAddr,
    pub t: Transcription,
    _scratch: Option<tempfile::TempDir>,
}

impl Server {
    /// Over the installed models, the config in memory.
    pub async fn over_models() -> Server {
        let opts = Options { models_dir: models_dir(), config_path: None, catalog: Catalog::new(None).unwrap(), mirror: None, release_api: None };
        Server::with(opts, None).await
    }

    /// Over a scratch directory (models and config), with `catalog` extra.
    pub async fn scratch(catalog: Option<&str>) -> Server {
        let dir = tempfile::tempdir().unwrap();
        let opts = Options {
            models_dir: dir.path().join("models"),
            config_path: Some(dir.path().join("transcription.toml")),
            catalog: Catalog::new(catalog).unwrap(),
            mirror: None,
            release_api: None,
        };
        Server::with(opts, Some(dir)).await
    }

    async fn with(opts: Options, scratch: Option<tempfile::TempDir>) -> Server {
        let t = Transcription::open(opts).unwrap();
        let args = LoginArgs { auth_provider: vec!["key".into()], admin: vec![ADMIN.into()], session_hours: 1 };
        let login = Login::wire(&args, &t, 0).unwrap();
        let serving = start(&login, "127.0.0.1:0".parse().unwrap()).await.unwrap();
        let addr = serving.addr;
        Server { login, serving: Some(serving), addr, t, _scratch: scratch }
    }

    pub fn scratch_path(&self) -> PathBuf {
        self._scratch.as_ref().unwrap().path().to_path_buf()
    }

    /// A signed-in caller's connection.
    pub async fn connect(&self, principal: &str) -> Client {
        let ctx = AuthContext::new(principal.into(), "sid".into(), vec![], json!({ "principal": principal, "agent": "test" }));
        let (token, _) = self.login.sessions().mint(ctx, 3600).unwrap();
        let ws = WsTransport::connect_with_headers(&format!("ws://{}", self.addr), "transcription", &[("Authorization".into(), format!("Bearer {token}"))])
            .await
            .unwrap();
        Client(Arc::new(ws))
    }

    /// `principal` holds `role` everywhere, granted as an admin does.
    pub async fn grant(&self, principal: &str, role: &str) {
        let admin = self.connect(ADMIN).await;
        let got = admin.call("auth", "grant", json!({ "principal": principal, "role": role, "scope": "*" })).await;
        assert!(got.iter().any(|e| e.data().is_some_and(|d| d["type"] == "granted")), "{got:?}");
    }

    pub async fn stop(mut self) {
        if let Some(s) = self.serving.take() {
            s.stop().await;
        }
    }
}

#[derive(Debug, Clone)]
pub enum Ev {
    Data(Value),
    /// A call failure: the error frame, as JSON.
    Failed(Value),
    Transport(String),
}

impl Ev {
    pub fn data(&self) -> Option<&Value> {
        match self {
            Ev::Data(v) => Some(v),
            _ => None,
        }
    }
}

fn ev(item: Result<StreamItem, plexus_rpc::client::TransportError>) -> Option<Ev> {
    match item {
        Ok(StreamItem::Data { content, .. }) => Some(Ev::Data(content)),
        Ok(e @ StreamItem::Error { .. }) => Some(Ev::Failed(serde_json::to_value(&e).unwrap())),
        Ok(_) => None,
        Err(e) => Some(Ev::Transport(e.to_string())),
    }
}

/// `namespace.method` on `path` (the activation's mount).
fn method(ns: &str, m: &str) -> MethodRef {
    MethodRef::new(Box::leak(ns.to_string().into_boxed_str()), Box::leak(m.to_string().into_boxed_str()))
}

#[derive(Clone)]
pub struct Client(pub Arc<WsTransport>);

impl Client {
    pub async fn call(&self, ns: &str, m: &str, params: Value) -> Vec<Ev> {
        self.0.call(method(ns, m), params).filter_map(|i| async { ev(i) }).collect().await
    }

    /// `speech.listen` with `audio` as its input; every event as it comes.
    pub fn listen<'a, S>(&'a self, params: Value, audio: S) -> impl Stream<Item = Ev> + 'a
    where
        S: Stream<Item = Bytes> + Send + 'a,
    {
        let input = Box::pin(audio.map(|b| Ok(InputItem::Binary(b))));
        self.0.call_with_input(method("speech", "listen"), params, input).filter_map(|i| async { ev(i) })
    }

    /// `speech.listen` over all of `pcm`, sent in `chunk`-byte frames.
    pub async fn listen_all(&self, params: Value, pcm: Vec<u8>, chunk: usize) -> Vec<Ev> {
        let chunks: Vec<Bytes> = pcm.chunks(chunk).map(Bytes::copy_from_slice).collect();
        self.listen(params, futures::stream::iter(chunks)).collect().await
    }
}

/// The finals' text, in order.
pub fn finals(evs: &[Ev]) -> Vec<String> {
    evs.iter().filter_map(|e| e.data()).filter(|d| d["type"] == "final").map(|d| d["text"].as_str().unwrap().to_string()).collect()
}

/// Word error rate of `hyp` against `reference`: case and punctuation aside.
pub fn wer(reference: &str, hyp: &str) -> f64 {
    let words = |s: &str| -> Vec<String> {
        s.to_lowercase().split(|c: char| c.is_whitespace()).map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>()).filter(|w| !w.is_empty()).collect()
    };
    let (r, h) = (words(reference), words(hyp));
    let mut d: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut prev = d[0];
        d[0] = i;
        for j in 1..=h.len() {
            let cur = d[j];
            d[j] = if r[i - 1] == h[j - 1] { prev } else { 1 + prev.min(d[j]).min(d[j - 1]) };
            prev = cur;
        }
    }
    d[h.len()] as f64 / r.len().max(1) as f64
}

/// This process's resident set, in MiB.
pub fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    let kb: f64 = s.lines().find(|l| l.starts_with("VmRSS:")).unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    kb / 1024.0
}

/// The error kind a failed call says.
pub fn kind(e: &Ev) -> String {
    match e {
        Ev::Failed(v) => v.to_string(),
        other => format!("{other:?}"),
    }
}
