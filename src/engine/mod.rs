//! What turns 16 kHz mono samples into text, behind a trait: sherpa-onnx on
//! the CPU today ([`sherpa`]); whisper.cpp or a GPU build can follow
//! (TRANSCRIBE-3) by implementing [`Engine`].
//!
//! An [`Engine`] is a loaded model, shared by every listener of it; a
//! [`Session`] is one listener's state over it (its stream, its voice
//! detector, its open utterance). A session's calls block — they run on the
//! decode pool, never on the socket's tasks.

pub mod sherpa;

use std::path::PathBuf;
use std::sync::Arc;

use crate::catalog::{Entry, Kind};
use crate::config::Resolved;

/// What a session says as audio goes in.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    /// The open utterance's text so far (it may change).
    Partial { text: String },
    /// An utterance, done: its text and where it was in the audio.
    Final { text: String, start_ms: u64, end_ms: u64 },
}

/// A loaded model.
pub trait Engine: Send + Sync + 'static {
    fn kind(&self) -> Kind;
    /// A new listener's state, under `settings`.
    fn open(&self, settings: &Resolved) -> anyhow::Result<Box<dyn Session>>;
}

/// One listener's state over an [`Engine`].
pub trait Session: Send + 'static {
    /// 16 kHz mono samples in; what they made out.
    fn accept(&mut self, samples: &[f32]) -> Vec<Piece>;
    /// The input ended: the open utterance, flushed as final.
    fn finish(&mut self) -> Vec<Piece>;
}

/// What loading a model needs.
#[derive(Debug, Clone)]
pub struct Load {
    pub entry: Entry,
    /// The model's installed directory.
    pub dir: PathBuf,
    /// The VAD's model file.
    pub vad: PathBuf,
    pub threads: u32,
}

/// Loads a model with the engine its catalog entry names.
pub fn load(l: &Load) -> anyhow::Result<Arc<dyn Engine>> {
    sherpa::load(l)
}

pub(crate) fn ms(samples: u64) -> u64 {
    samples * 1000 / crate::audio::RATE as u64
}
