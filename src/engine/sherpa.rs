//! sherpa-onnx (k2-fsa's own Rust API, onnxruntime on the CPU) with Silero
//! VAD — the engine manifest's Node transcriber runs, here in-process.
//!
//! - **Streaming** (Nemotron, zipformer): every sample goes to the
//!   recognizer's stream; its text past what was already final is the
//!   `partial`. The VAD cuts utterances: once a segment has ended and the
//!   recognizer has been fed `final_tail_ms` past its end (its look-ahead),
//!   the text since the last final is the segment's `final`. One stream per
//!   listener lives across utterances; after 1.5 s of quiet with nothing
//!   open it is reset, so it never grows without bound in a long call.
//! - **Offline** (Parakeet TDT, Moonshine, Whisper): each VAD segment is
//!   decoded whole for its `final`; while speech is open, the utterance so
//!   far is re-decoded every `partial_ms` of new audio for a `partial`.
//!
//! One recognizer is loaded per model and shared: onnxruntime's `Run` is
//! safe to call from many threads, and each listener has its own stream.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, SileroVadModelConfig,
    VadModelConfig, VoiceActivityDetector,
};

use super::{ms, Engine, Load, Piece, Session};
use crate::audio::RATE;
use crate::catalog::{Family, Kind};
use crate::config::{Resolved, Vad};

/// Silero's window, in samples at 16 kHz.
const WINDOW: usize = 512;
/// Quiet after which a streaming listener's stream is reset (samples).
const QUIET_RESET: u64 = RATE as u64 * 3 / 2;

/// The file in `dir` whose name starts with `prefix` and ends `.onnx`,
/// preferring an int8 one.
fn pick(dir: &Path, prefix: &str) -> anyhow::Result<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("{}", dir.display()))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".onnx"))
        .collect();
    names.sort_by_key(|n| (!n.contains("int8"), n.len()));
    let name = names.first().with_context(|| format!("{}: no {prefix}*.onnx", dir.display()))?;
    Ok(dir.join(name).to_string_lossy().into_owned())
}

fn path(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// The files a model of `family` loads, as they are in its directory —
/// what a download checks before it calls the model installed.
pub fn check_files(family: Family, dir: &Path) -> anyhow::Result<()> {
    match family {
        Family::OnlineTransducer | Family::NemoTransducer => {
            for p in ["encoder", "decoder", "joiner"] {
                pick(dir, p)?;
            }
        }
        Family::Moonshine => {
            for p in ["preprocess", "encode", "uncached_decode", "cached_decode"] {
                pick(dir, p)?;
            }
        }
        Family::Whisper => {
            whisper_file(dir, "encoder")?;
            whisper_file(dir, "decoder")?;
        }
        Family::SileroVad => return Ok(()),
    }
    anyhow::ensure!(dir.join(tokens_name(family, dir)).is_file(), "{}: no tokens file", dir.display());
    Ok(())
}

fn tokens_name(family: Family, dir: &Path) -> String {
    if family == Family::Whisper {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.ends_with("-tokens.txt") {
                    return n;
                }
            }
        }
    }
    "tokens.txt".into()
}

/// Whisper's archives name files `<size>-encoder.int8.onnx`.
fn whisper_file(dir: &Path, part: &str) -> anyhow::Result<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.contains(&format!("-{part}")) && n.ends_with(".onnx"))
        .collect();
    names.sort_by_key(|n| (!n.contains("int8"), n.len()));
    let name = names.first().with_context(|| format!("{}: no *-{part}*.onnx", dir.display()))?;
    Ok(dir.join(name).to_string_lossy().into_owned())
}

pub fn load(l: &Load) -> anyhow::Result<Arc<dyn Engine>> {
    let dir = &l.dir;
    let tokens = Some(path(&dir.join(tokens_name(l.entry.family, dir))));
    let threads = l.threads as i32;
    let vad = l.vad.clone();
    match (l.entry.kind, l.entry.family) {
        (Kind::Streaming, Family::OnlineTransducer) => {
            let mut c = OnlineRecognizerConfig::default();
            c.model_config.transducer.encoder = Some(pick(dir, "encoder")?);
            c.model_config.transducer.decoder = Some(pick(dir, "decoder")?);
            c.model_config.transducer.joiner = Some(pick(dir, "joiner")?);
            c.model_config.tokens = tokens;
            c.model_config.num_threads = threads;
            c.model_config.provider = Some("cpu".into());
            c.decoding_method = Some("greedy_search".into());
            let rec = OnlineRecognizer::create(&c).with_context(|| format!("{}: the recognizer did not load", l.entry.id))?;
            Ok(Arc::new(Online { rec: Arc::new(rec), vad }))
        }
        (Kind::Offline, family) => {
            let mut c = OfflineRecognizerConfig::default();
            match family {
                Family::NemoTransducer => {
                    c.model_config.transducer.encoder = Some(pick(dir, "encoder")?);
                    c.model_config.transducer.decoder = Some(pick(dir, "decoder")?);
                    c.model_config.transducer.joiner = Some(pick(dir, "joiner")?);
                    c.model_config.model_type = Some("nemo_transducer".into());
                }
                Family::Moonshine => {
                    c.model_config.moonshine.preprocessor = Some(pick(dir, "preprocess")?);
                    c.model_config.moonshine.encoder = Some(pick(dir, "encode")?);
                    c.model_config.moonshine.uncached_decoder = Some(pick(dir, "uncached_decode")?);
                    c.model_config.moonshine.cached_decoder = Some(pick(dir, "cached_decode")?);
                }
                Family::Whisper => {
                    c.model_config.whisper.encoder = Some(whisper_file(dir, "encoder")?);
                    c.model_config.whisper.decoder = Some(whisper_file(dir, "decoder")?);
                    c.model_config.whisper.language = Some("en".into());
                    c.model_config.whisper.task = Some("transcribe".into());
                    c.model_config.whisper.tail_paddings = -1;
                }
                other => anyhow::bail!("{}: {other:?} is not an offline family", l.entry.id),
            }
            c.model_config.tokens = tokens;
            c.model_config.num_threads = threads;
            c.model_config.provider = Some("cpu".into());
            c.decoding_method = Some("greedy_search".into());
            let rec = OfflineRecognizer::create(&c).with_context(|| format!("{}: the recognizer did not load", l.entry.id))?;
            Ok(Arc::new(Offline { rec: Arc::new(rec), vad }))
        }
        (kind, family) => anyhow::bail!("{}: no engine runs a {kind:?} {family:?} model", l.entry.id),
    }
}

fn detector(model: &Path, v: &Vad) -> anyhow::Result<VoiceActivityDetector> {
    let c = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(path(model)),
            threshold: v.threshold,
            min_silence_duration: v.min_silence_ms as f32 / 1000.0,
            min_speech_duration: v.min_speech_ms as f32 / 1000.0,
            window_size: WINDOW as i32,
            max_speech_duration: v.max_speech_s,
        },
        sample_rate: RATE as i32,
        num_threads: 1,
        provider: Some("cpu".into()),
        ..Default::default()
    };
    VoiceActivityDetector::create(&c, v.max_speech_s + 10.0).context("the voice activity detector did not load")
}

/// Feeds Silero whole windows, keeping the remainder.
struct Windows {
    vad: VoiceActivityDetector,
    rest: Vec<f32>,
    /// Samples the detector has been fed.
    fed: u64,
}

impl Windows {
    /// Feeds `samples`; calls `voice(at)` for each window that is speech.
    fn feed(&mut self, samples: &[f32], mut voice: impl FnMut(u64)) {
        self.rest.extend_from_slice(samples);
        let whole = self.rest.len() / WINDOW * WINDOW;
        for w in self.rest[..whole].chunks(WINDOW) {
            self.vad.accept_waveform(w);
            self.fed += WINDOW as u64;
            if self.vad.detected() {
                voice(self.fed);
            }
        }
        self.rest.drain(..whole);
    }

    /// Every segment ended so far: (start, end) in samples, and its audio.
    fn segments(&mut self) -> Vec<(u64, u64, Vec<f32>)> {
        let mut out = Vec::new();
        while let Some(seg) = self.vad.front() {
            let start = seg.start().max(0) as u64;
            out.push((start, start + seg.n().max(0) as u64, seg.samples().to_vec()));
            self.vad.pop();
        }
        out
    }

    fn flush(&mut self) {
        if !self.rest.is_empty() {
            let mut w = std::mem::take(&mut self.rest);
            w.resize(WINDOW, 0.0);
            self.vad.accept_waveform(&w);
        }
        self.vad.flush();
    }
}

fn windows(model: &Path, v: &Vad) -> anyhow::Result<Windows> {
    Ok(Windows { vad: detector(model, v)?, rest: Vec::new(), fed: 0 })
}

// ── streaming ────────────────────────────────────────────────────────────

struct Online {
    rec: Arc<OnlineRecognizer>,
    vad: std::path::PathBuf,
}

impl Engine for Online {
    fn kind(&self) -> Kind {
        Kind::Streaming
    }

    fn open(&self, s: &Resolved) -> anyhow::Result<Box<dyn Session>> {
        Ok(Box::new(OnlineSession {
            stream: self.rec.create_stream(),
            rec: self.rec.clone(),
            vad: windows(&self.vad, &s.vad)?,
            committed: 0,
            partial: String::new(),
            tail: s.final_tail_ms as usize * RATE as usize / 1000,
            last_voice: 0,
            last_end: 0,
        }))
    }
}

/// One listener of a streaming model: one stream across its utterances.
///
/// A streaming model decodes much better with its left context (Nemotron
/// mis-hears words seconds into an utterance on a stream begun at it), so
/// the stream runs the whole call. Silero ends a segment only once it has
/// heard `min_silence` of silence, so at that moment everything the stream
/// has been fed past the segment is silence and the next utterance has not
/// begun: the stream is given `final_tail_ms` more silence there and then
/// (its look-ahead, so the last words come out), and every token since the
/// last final is this one's. Nothing of the next utterance can be in it, and
/// the final is said `min_silence` after the speech ends.
struct OnlineSession {
    rec: Arc<OnlineRecognizer>,
    stream: OnlineStream,
    vad: Windows,
    /// Tokens of the stream already said as final.
    committed: usize,
    partial: String,
    /// Silence given to flush the look-ahead (samples).
    tail: usize,
    last_voice: u64,
    last_end: u64,
}

/// A token that is punctuation alone.
fn punct(t: &str) -> bool {
    let t = t.trim();
    !t.is_empty() && t.chars().all(|c| c.is_ascii_punctuation())
}

/// Tokens as text; punctuation at the head (an utterance's end the model
/// stamped late) is not the next utterance's.
fn join(tokens: &[String]) -> String {
    let mut tokens = tokens;
    while tokens.first().is_some_and(|t| punct(t)) {
        tokens = &tokens[1..];
    }
    tokens.concat().replace('\u{2581}', " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

impl OnlineSession {
    fn decode(&self) {
        while self.rec.is_ready(&self.stream) {
            self.rec.decode(&self.stream);
        }
    }

    fn tokens(&self) -> Vec<String> {
        self.rec.get_result(&self.stream).map(|r| r.tokens).unwrap_or_default()
    }

    fn open_text(&self) -> String {
        let t = self.tokens();
        join(&t[self.committed.min(t.len())..])
    }

    /// The tokens since the last final, as the final of `start..end`.
    fn commit(&mut self, start: u64, end: u64, out: &mut Vec<Piece>) {
        let t = self.tokens();
        let text = join(&t[self.committed.min(t.len())..]);
        self.committed = t.len();
        self.partial.clear();
        self.last_end = end;
        if !text.is_empty() {
            out.push(Piece::Final { text, start_ms: ms(start), end_ms: ms(end.max(start)) });
        }
    }
}

impl Session for OnlineSession {
    fn accept(&mut self, samples: &[f32]) -> Vec<Piece> {
        let mut out = Vec::new();
        // Window by window, so a segment's end is acted on where the VAD
        // calls it, before any audio after it reaches the stream.
        for w in samples.chunks(WINDOW) {
            let mut voice = self.last_voice;
            self.vad.feed(w, |at| voice = at);
            self.last_voice = voice;
            self.stream.accept_waveform(RATE as i32, w);
            let segments = self.vad.segments();
            if let (Some(first), Some(last)) = (segments.first(), segments.last()) {
                let (start, end) = (first.0, last.1);
                self.stream.accept_waveform(RATE as i32, &vec![0.0; self.tail]);
                self.decode();
                self.commit(start, end, &mut out);
            }
        }
        self.decode();
        let open = self.open_text();
        if !open.is_empty() && open != self.partial {
            self.partial = open.clone();
            out.push(Piece::Partial { text: open.clone() });
        }
        // A long quiet with nothing open: start the stream afresh, so it
        // never grows without bound in a long call.
        if open.is_empty() && self.vad.fed.saturating_sub(self.last_voice) >= QUIET_RESET && self.committed > 0 {
            self.rec.reset(&self.stream);
            self.committed = 0;
        }
        out
    }

    fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        self.vad.flush();
        let segments = self.vad.segments();
        self.stream.accept_waveform(RATE as i32, &vec![0.0; self.tail.max(RATE as usize / 2) + RATE as usize / 4]);
        self.stream.input_finished();
        self.decode();
        // The open utterance, or words the VAD did not call speech.
        let start = segments.first().map(|s| s.0).unwrap_or(self.last_end);
        let end = segments.last().map(|s| s.1).unwrap_or(self.vad.fed);
        self.commit(start, end, &mut out);
        out
    }
}

// ── offline ──────────────────────────────────────────────────────────────

struct Offline {
    rec: Arc<OfflineRecognizer>,
    vad: std::path::PathBuf,
}

impl Engine for Offline {
    fn kind(&self) -> Kind {
        Kind::Offline
    }

    fn open(&self, s: &Resolved) -> anyhow::Result<Box<dyn Session>> {
        // Silero says "speech" this long after it began: look back that far.
        let lookback = (s.vad.min_speech_ms as u64 + 300) * RATE as u64 / 1000;
        Ok(Box::new(OfflineSession {
            rec: self.rec.clone(),
            vad: windows(&self.vad, &s.vad)?,
            buf: VecDeque::new(),
            buf_start: 0,
            fed: 0,
            speaking: false,
            speech_from: 0,
            since_partial: 0,
            partial_every: s.partial_ms as u64 * RATE as u64 / 1000,
            lookback,
            keep: (s.vad.max_speech_s as u64 + 2) * RATE as u64,
            partial: String::new(),
        }))
    }
}

struct OfflineSession {
    rec: Arc<OfflineRecognizer>,
    vad: Windows,
    /// The recent audio (for partials), from `buf_start`.
    buf: VecDeque<f32>,
    buf_start: u64,
    fed: u64,
    speaking: bool,
    speech_from: u64,
    since_partial: u64,
    partial_every: u64,
    lookback: u64,
    keep: u64,
    partial: String,
}

impl OfflineSession {
    fn decode(&self, samples: &[f32]) -> String {
        let st = self.rec.create_stream();
        st.accept_waveform(RATE as i32, samples);
        self.rec.decode(&st);
        st.get_result().map(|r| r.text.trim().to_string()).unwrap_or_default()
    }

    fn finals(&mut self, out: &mut Vec<Piece>) {
        for (start, end, samples) in self.vad.segments() {
            let text = self.decode(&samples);
            self.partial.clear();
            self.since_partial = 0;
            self.speech_from = end;
            if !text.is_empty() {
                out.push(Piece::Final { text, start_ms: ms(start), end_ms: ms(end) });
            }
        }
    }
}

impl Session for OfflineSession {
    fn accept(&mut self, samples: &[f32]) -> Vec<Piece> {
        let mut out = Vec::new();
        self.buf.extend(samples.iter().copied());
        self.fed += samples.len() as u64;
        let mut first_voice = None;
        self.vad.feed(samples, |at| {
            first_voice.get_or_insert(at);
        });
        let was = self.speaking;
        self.speaking = self.vad.vad.detected();
        if let (false, Some(at)) = (was, first_voice) {
            self.speech_from = self.speech_from.max(at.saturating_sub(self.lookback));
            self.since_partial = 0;
        }
        self.finals(&mut out);
        if self.speaking && self.partial_every > 0 {
            self.since_partial += samples.len() as u64;
            if self.since_partial >= self.partial_every {
                self.since_partial = 0;
                let from = self.speech_from.max(self.buf_start) - self.buf_start;
                let audio: Vec<f32> = self.buf.iter().skip(from as usize).copied().collect();
                let text = self.decode(&audio);
                if !text.is_empty() && text != self.partial {
                    self.partial = text.clone();
                    out.push(Piece::Partial { text });
                }
            }
        }
        // Keep only what a partial could still need.
        let keep = if self.speaking { self.keep } else { self.lookback + RATE as u64 };
        while self.buf.len() as u64 > keep {
            self.buf.pop_front();
            self.buf_start += 1;
        }
        out
    }

    fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        self.vad.flush();
        self.finals(&mut out);
        out
    }
}
