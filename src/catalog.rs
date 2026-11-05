//! The models this service knows: what each is, where its archive is, how to
//! check it, and which files in it the engine loads.
//!
//! Every built-in entry is an asset of the sherpa-onnx GitHub release
//! `asr-models` (k2-fsa/sherpa-onnx). `--catalog FILE` adds entries (or
//! replaces one by id) from a TOML file of `[[model]]` tables, the same
//! fields as [`Entry`] — how a model the list does not carry, or a mirror, is
//! used.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Where the built-in archives are downloaded from.
pub const RELEASE_DOWNLOAD: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";
/// The release's description, read by `models search`.
pub const RELEASE_API: &str = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/asr-models";

/// The voice activity detector every `listen` runs, downloaded with the first
/// model.
pub const VAD_ID: &str = "silero-vad";

/// How a model decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Online: text grows as audio arrives (partials from the recognizer).
    Streaming,
    /// Offline: an utterance is decoded whole (partials by re-decoding it).
    Offline,
    /// Not a recognizer: the voice activity detector.
    Vad,
}

/// Which sherpa-onnx model family an entry is — what files it loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// An online transducer (zipformer, Nemotron): encoder, decoder, joiner.
    OnlineTransducer,
    /// An offline NeMo transducer (Parakeet TDT): encoder, decoder, joiner.
    NemoTransducer,
    /// Moonshine: preprocessor, encoder, uncached and cached decoders.
    Moonshine,
    /// Whisper: encoder, decoder.
    Whisper,
    /// Silero VAD: one `.onnx` file.
    SileroVad,
}

/// One model the service can download and run.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// The id `listen --model` and `models download --id` take.
    pub id: String,
    /// What it is, in a line.
    pub description: String,
    pub kind: Kind,
    pub family: Family,
    /// Languages, ISO 639-1.
    pub languages: Vec<String>,
    /// The release asset: a `.tar.bz2` holding one directory, or a lone
    /// `.onnx` (the VAD).
    pub archive: String,
    /// Its URL; default `RELEASE_DOWNLOAD/archive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Its size in bytes: checked after the download.
    pub size: u64,
    /// Its sha256, hex: checked when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Where `sha256` comes from: `github` (the release's own digest) or
    /// `measured` (taken from a download here, trust on first use).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256_source: Option<String>,
}

impl Entry {
    pub fn url(&self, mirror: Option<&str>) -> String {
        match (&self.url, mirror) {
            (Some(u), _) => u.clone(),
            (None, Some(m)) => format!("{}/{}", m.trim_end_matches('/'), self.archive),
            (None, None) => format!("{RELEASE_DOWNLOAD}/{}", self.archive),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn e(id: &str, description: &str, kind: Kind, family: Family, languages: &[&str], archive: &str, size: u64, sha256: Option<(&str, &str)>) -> Entry {
    Entry {
        id: id.into(),
        description: description.into(),
        kind,
        family,
        languages: languages.iter().map(|l| l.to_string()).collect(),
        archive: archive.into(),
        url: None,
        size,
        sha256: sha256.map(|(h, _)| h.into()),
        sha256_source: sha256.map(|(_, s)| s.into()),
    }
}

/// The 25 languages Parakeet TDT 0.6B v3 transcribes.
const EU25: &[&str] = &["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"];

/// The built-in catalog.
pub fn builtin() -> Vec<Entry> {
    use Family::*;
    use Kind::*;
    vec![
        e("nemotron-en-560ms", "NVIDIA Nemotron Speech Streaming 0.6B, English, 560 ms chunks, int8 (manifest's engine)", Streaming, OnlineTransducer, &["en"],
          "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2", 463_945_051,
          Some(("78e2b79fcf7271553a74402a76b771b09ea40117a39566a79f52235b23db6358", "github"))),
        e("nemotron-en-160ms", "NVIDIA Nemotron Speech Streaming 0.6B, English, 160 ms chunks, int8", Streaming, OnlineTransducer, &["en"],
          "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2", 463_945_198,
          Some(("0ae73a41cd51599dc7cac9ac083d9d35de53d762ca45923505fde47a3751814b", "github"))),
        e("nemotron-en-80ms", "NVIDIA Nemotron Speech Streaming 0.6B, English, 80 ms chunks, int8", Streaming, OnlineTransducer, &["en"],
          "sherpa-onnx-nemotron-speech-streaming-en-0.6b-80ms-int8-2026-04-25.tar.bz2", 463_945_379,
          Some(("caaf92069dbd1ca054f8e17cab179813bc28b4585f5c392540357ece4722333d", "github"))),
        e("zipformer-en-20m", "k2 streaming zipformer, English, 20M parameters, int8 — small and fast, lower accuracy", Streaming, OnlineTransducer, &["en"],
          "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2", 127_887_156,
          Some(("9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9", "measured"))),
        e("parakeet-tdt-v2", "NVIDIA Parakeet TDT 0.6B v2, English, int8 — offline, punctuated", Offline, NemoTransducer, &["en"],
          "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2", 482_468_385,
          Some(("157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad", "github"))),
        e("parakeet-tdt-v3", "NVIDIA Parakeet TDT 0.6B v3, 25 European languages, int8 — offline, punctuated", Offline, NemoTransducer, EU25,
          "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2", 487_170_055,
          Some(("5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf", "github"))),
        e("moonshine-tiny-en", "Useful Sensors Moonshine tiny, English, int8 — offline, very fast", Offline, Moonshine, &["en"],
          "sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", 107_600_538,
          Some(("d5fe6ec4334fef36255b2a4010412cad4c007e33103fec62fb5d17cad88086f2", "measured"))),
        e("moonshine-base-en", "Useful Sensors Moonshine base, English, int8 — offline", Offline, Moonshine, &["en"],
          "sherpa-onnx-moonshine-base-en-int8.tar.bz2", 250_807_309, None),
        e("whisper-tiny-en", "OpenAI Whisper tiny.en — offline", Offline, Whisper, &["en"],
          "sherpa-onnx-whisper-tiny.en.tar.bz2", 118_071_777, None),
        e(VAD_ID, "Silero VAD — the utterance boundaries every listen uses (fetched with the first model)", Vad, SileroVad, &[],
          "silero_vad.onnx", 643_854,
          Some(("9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6", "github"))),
    ]
}

/// The catalog: the built-ins, then `--catalog`'s entries over them by id.
#[derive(Debug, Clone)]
pub struct Catalog {
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Extra {
    #[serde(default)]
    model: Vec<Entry>,
}

impl Catalog {
    pub fn new(extra: Option<&str>) -> anyhow::Result<Self> {
        let mut entries = builtin();
        if let Some(text) = extra {
            let extra: Extra = toml::from_str(text).map_err(|e| anyhow::anyhow!("--catalog: {e}"))?;
            for m in extra.model {
                check_id(&m.id).map_err(|e| anyhow::anyhow!("--catalog: {e}"))?;
                match entries.iter_mut().find(|x| x.id == m.id) {
                    Some(slot) => *slot = m,
                    None => entries.push(m),
                }
            }
        }
        Ok(Self { entries })
    }

    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Recognizers only (not the VAD).
    pub fn models(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|e| e.kind != Kind::Vad)
    }

    pub fn ids(&self) -> String {
        self.models().map(|e| e.id.as_str()).collect::<Vec<_>>().join(", ")
    }
}

/// An id is a directory name under the models directory: `[a-z0-9-]`.
pub fn check_id(id: &str) -> Result<(), String> {
    // No `.`: an id is a bare key of the config's `[models.<id>]`.
    if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') {
        return Err(format!("`{id}` is not a model id (lower-case letters, digits and `-`)"));
    }
    Ok(())
}

/// Does `q` match `text`? Every word of `q`, case-insensitive, `-`/`_`/`.` as spaces.
pub fn matches(q: &str, text: &str) -> bool {
    let norm = |s: &str| s.to_ascii_lowercase().replace(['-', '_', '.'], " ");
    let hay = norm(text);
    norm(q).split_whitespace().all(|w| hay.contains(w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_valid() {
        let c = Catalog::new(None).unwrap();
        let mut ids: Vec<_> = c.entries().iter().map(|e| e.id.clone()).collect();
        ids.iter().for_each(|i| check_id(i).unwrap());
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), c.entries().len());
    }

    #[test]
    fn an_extra_catalog_adds_and_replaces() {
        let c = Catalog::new(Some(
            r#"[[model]]
id = "fixture"
description = "a fixture"
kind = "streaming"
family = "online_transducer"
languages = ["en"]
archive = "fixture.tar.bz2"
url = "http://127.0.0.1:1/fixture.tar.bz2"
size = 10
"#,
        ))
        .unwrap();
        assert_eq!(c.get("fixture").unwrap().url(None), "http://127.0.0.1:1/fixture.tar.bz2");
        assert!(Catalog::new(Some("[[model]]\nid = \"x\"\nbogus = 1\n")).is_err());
    }

    #[test]
    fn search_words() {
        assert!(matches("parakeet", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"));
        assert!(matches("parakeet v3", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"));
        assert!(!matches("parakeet v4", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"));
    }
}
