//! `transcription.toml`: what `models config --config @file.toml` sets.
//!
//! Every key is known (an unknown one is refused, named); a value out of its
//! range is refused, named; the file is kept in the data directory and read
//! at start. A `listen` reads the config when it begins, so a change applies
//! to the next one and never to a call under way.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::catalog::{Catalog, Kind};

/// The whole file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The model `listen` runs when `--model` is not given.
    #[serde(default = "d_model")]
    pub default_model: String,
    /// Threads one decode uses (onnxruntime's intra-op threads).
    #[serde(default = "d_threads")]
    pub threads: u32,
    /// Decodes that may run at once across every listener (the pool's size);
    /// a listener past it waits, and its audio waits in the client.
    #[serde(default = "d_workers")]
    pub decode_workers: u32,
    /// For an offline model: re-decode the open utterance for a `partial`
    /// after this much new audio (ms). 0: no partials.
    #[serde(default = "d_partial")]
    pub partial_ms: u32,
    /// The voice activity detector.
    #[serde(default)]
    pub vad: Vad,
    /// Per-model overrides, `[models.<id>]`.
    #[serde(default)]
    pub models: BTreeMap<String, ModelOverride>,
}

/// `[vad]`: how utterances are cut.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vad {
    /// Speech probability above which a window is speech (0..1).
    #[serde(default = "d_vad_threshold")]
    pub threshold: f32,
    /// Silence that ends an utterance (ms).
    #[serde(default = "d_min_silence")]
    pub min_silence_ms: u32,
    /// Speech shorter than this is not an utterance (ms).
    #[serde(default = "d_min_speech")]
    pub min_speech_ms: u32,
    /// An utterance longer than this is cut (s).
    #[serde(default = "d_max_speech")]
    pub max_speech_s: f32,
}

/// `[models.<id>]`: what differs for one model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_ms: Option<u32>,
    /// For a streaming model: silence fed to finish an utterance's stream
    /// (ms), so its look-ahead lets the last words out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_tail_ms: Option<u32>,
}

fn d_model() -> String {
    "nemotron-en-560ms".into()
}
fn d_threads() -> u32 {
    2
}
fn d_workers() -> u32 {
    let n = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);
    (n / d_threads()).clamp(1, 16)
}
fn d_partial() -> u32 {
    700
}
fn d_vad_threshold() -> f32 {
    0.5
}
fn d_min_silence() -> u32 {
    400
}
fn d_min_speech() -> u32 {
    150
}
fn d_max_speech() -> f32 {
    20.0
}

impl Default for Vad {
    fn default() -> Self {
        Vad { threshold: d_vad_threshold(), min_silence_ms: d_min_silence(), min_speech_ms: d_min_speech(), max_speech_s: d_max_speech() }
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            default_model: d_model(),
            threads: d_threads(),
            decode_workers: d_workers(),
            partial_ms: d_partial(),
            vad: Vad::default(),
            models: BTreeMap::new(),
        }
    }
}

/// What one `listen` runs with: the config, resolved for its model.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub threads: u32,
    pub partial_ms: u32,
    pub final_tail_ms: u32,
    pub vad: Vad,
}

impl Config {
    /// Parses and checks a file's text against `catalog`.
    pub fn parse(text: &str, catalog: &Catalog) -> Result<Self, String> {
        let c: Config = toml::from_str(text).map_err(|e| e.message().to_string()).map_err(|m| format!("the config is refused: {m}"))?;
        c.check(catalog)?;
        Ok(c)
    }

    pub fn check(&self, catalog: &Catalog) -> Result<(), String> {
        let model = |id: &str, key: &str| match catalog.get(id) {
            Some(e) if e.kind != Kind::Vad => Ok(()),
            _ => Err(format!("the config is refused: {key} = \"{id}\" is no model (the models: {})", catalog.ids())),
        };
        model(&self.default_model, "default_model")?;
        let range = |key: &str, v: u32, lo: u32, hi: u32| {
            if (lo..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("the config is refused: {key} = {v} is outside {lo}..={hi}"))
            }
        };
        range("threads", self.threads, 1, 64)?;
        range("decode_workers", self.decode_workers, 1, 256)?;
        range("partial_ms", self.partial_ms, 0, 60_000)?;
        if !(0.01..=0.99).contains(&self.vad.threshold) {
            return Err(format!("the config is refused: vad.threshold = {} is outside 0.01..=0.99", self.vad.threshold));
        }
        range("vad.min_silence_ms", self.vad.min_silence_ms, 50, 10_000)?;
        range("vad.min_speech_ms", self.vad.min_speech_ms, 10, 10_000)?;
        if !(1.0..=120.0).contains(&self.vad.max_speech_s) {
            return Err(format!("the config is refused: vad.max_speech_s = {} is outside 1..=120", self.vad.max_speech_s));
        }
        for (id, o) in &self.models {
            model(id, &format!("[models.{id}]"))?;
            if let Some(t) = o.threads {
                range(&format!("models.{id}.threads"), t, 1, 64)?;
            }
            if let Some(p) = o.partial_ms {
                range(&format!("models.{id}.partial_ms"), p, 0, 60_000)?;
            }
            if let Some(t) = o.final_tail_ms {
                range(&format!("models.{id}.final_tail_ms"), t, 0, 5_000)?;
            }
        }
        Ok(())
    }

    /// The settings for `model` (an id the catalog holds).
    pub fn resolve(&self, model: &str, kind: Kind) -> Resolved {
        let o = self.models.get(model).cloned().unwrap_or_default();
        // A streaming model's text trails its audio by about one chunk.
        let tail = if model.ends_with("560ms") { 1200 } else if kind == Kind::Streaming { 500 } else { 0 };
        Resolved {
            threads: o.threads.unwrap_or(self.threads),
            partial_ms: o.partial_ms.unwrap_or(self.partial_ms),
            final_tail_ms: o.final_tail_ms.unwrap_or(tail),
            vad: self.vad.clone(),
        }
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("a config serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat() -> Catalog {
        Catalog::new(None).unwrap()
    }

    #[test]
    fn the_example_file_parses() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/transcription.toml")).unwrap();
        let c = Config::parse(&text, &cat()).unwrap();
        assert_eq!(c.default_model, "nemotron-en-560ms");
        assert_eq!(c.models["parakeet-tdt-v2"].partial_ms, Some(800));
    }

    #[test]
    fn an_unknown_key_is_named() {
        let e = Config::parse("default_model = \"nemotron-en-560ms\"\nthreds = 4\n", &cat()).unwrap_err();
        assert!(e.contains("threds"), "{e}");
        let e = Config::parse("[vad]\nthreshhold = 0.4\n", &cat()).unwrap_err();
        assert!(e.contains("threshhold"), "{e}");
        let e = Config::parse("[models.parakeet-tdt-v2]\npartial = 3\n", &cat()).unwrap_err();
        assert!(e.contains("partial"), "{e}");
    }

    #[test]
    fn values_are_checked() {
        assert!(Config::parse("default_model = \"silero-vad\"\n", &cat()).unwrap_err().contains("no model"));
        assert!(Config::parse("threads = 0\n", &cat()).unwrap_err().contains("threads"));
        assert!(Config::parse("[models.nope]\nthreads = 2\n", &cat()).unwrap_err().contains("nope"));
        assert_eq!(Config::parse("", &cat()).unwrap(), Config::default());
    }

    #[test]
    fn round_trips() {
        let c = Config::default();
        assert_eq!(Config::parse(&c.to_toml(), &cat()).unwrap(), c);
    }
}
