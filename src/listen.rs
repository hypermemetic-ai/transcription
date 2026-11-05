//! `listen`'s pipeline: the caller's PCM in, text out, on one call.
//!
//! The handler reads a chunk, takes what else has already arrived (up to
//! [`BATCH`]), converts it to 16 kHz mono and hands it to its session on the
//! decode pool. It reads nothing more until that decode is back, so a
//! listener whose decode falls behind consumes slower, the input half grants
//! less credit, and the client holds its audio — nothing is dropped, and the
//! socket's reader never waits on a decode.

use std::sync::Arc;

use bytes::Bytes;
use futures::{FutureExt, Stream, StreamExt};
use plexus_core::plexus::Input;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::audio::{Converter, Format, Shape};
use crate::engine::{Piece, Session};
use crate::store::{not_downloaded, Store};

/// What `listen` streams back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ListenEvent {
    /// The open utterance's text so far; a later partial or the final
    /// replaces it.
    Partial { text: String },
    /// An utterance, done (VAD-segmented): its text and its place in the
    /// audio, in ms from the call's first sample.
    Final { text: String, start_ms: u64, end_ms: u64 },
    /// Refused, in band (no such model, not downloaded, a bad shape).
    Error { message: String },
}

impl plexus_client::FromClientError for ListenEvent {
    fn from_client_error(error: plexus_client::ClientError) -> Self {
        ListenEvent::Error { message: error.to_string() }
    }
}

impl From<Piece> for ListenEvent {
    fn from(p: Piece) -> Self {
        match p {
            Piece::Partial { text } => ListenEvent::Partial { text },
            Piece::Final { text, start_ms, end_ms } => ListenEvent::Final { text, start_ms, end_ms },
        }
    }
}

/// The most input taken into one decode, in bytes of the caller's PCM.
const BATCH: usize = 64 * 1024;

/// What a `listen` call asked for.
#[derive(Debug, Clone, Default)]
pub struct Ask {
    pub model: Option<String>,
    pub rate: Option<u32>,
    pub format: Option<Format>,
    pub channels: Option<u16>,
}

pub fn listen(store: Arc<Store>, ask: Ask, audio: Input<Bytes>) -> impl Stream<Item = ListenEvent> + Send + 'static {
    async_stream::stream! {
        let config = store.config();
        let id = ask.model.unwrap_or_else(|| config.default_model.clone());
        let entry = match store.recognizer(&id) {
            Ok(e) => e.clone(),
            Err(message) => {
                yield ListenEvent::Error { message };
                return;
            }
        };
        if store.installed(&id).is_none() {
            yield ListenEvent::Error { message: not_downloaded(&id) };
            return;
        }
        if store.installed(crate::catalog::VAD_ID).is_none() {
            yield ListenEvent::Error { message: not_downloaded(crate::catalog::VAD_ID) };
            return;
        }
        let shape = Shape { rate: ask.rate.unwrap_or(crate::audio::RATE), format: ask.format.unwrap_or_default(), channels: ask.channels.unwrap_or(1) };
        let mut conv = match Converter::new(shape) {
            Ok(c) => c,
            Err(message) => {
                yield ListenEvent::Error { message };
                return;
            }
        };
        let settings = config.resolve(&id, entry.kind);
        let engine = match store.engine(&id, settings.threads).await {
            Ok(e) => e,
            Err(message) => {
                yield ListenEvent::Error { message };
                return;
            }
        };
        let opened = tokio::task::spawn_blocking(move || engine.open(&settings)).await;
        let mut session: Box<dyn Session> = match opened {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                yield ListenEvent::Error { message: format!("{e:#}") };
                return;
            }
            Err(e) => {
                yield ListenEvent::Error { message: e.to_string() };
                return;
            }
        };
        let mut audio = audio.into_stream();
        let mut ended = false;
        while !ended {
            let Some(first) = audio.next().await else { break };
            let mut pcm = first.to_vec();
            // What else is here already, so a burst is one decode.
            while pcm.len() < BATCH {
                match audio.next().now_or_never() {
                    Some(Some(more)) => pcm.extend_from_slice(&more),
                    Some(None) => {
                        ended = true;
                        break;
                    }
                    None => break,
                }
            }
            let samples = conv.push(&pcm);
            if samples.is_empty() {
                continue;
            }
            let permit = store.pool().acquire_owned().await.expect("the pool is never closed");
            let done = tokio::task::spawn_blocking(move || {
                let pieces = session.accept(&samples);
                drop(permit);
                (session, pieces)
            })
            .await;
            let (s, pieces) = match done {
                Ok(d) => d,
                Err(e) => {
                    yield ListenEvent::Error { message: format!("decode: {e}") };
                    return;
                }
            };
            session = s;
            for p in pieces {
                yield p.into();
            }
        }
        // The caller ended its input: the rest, and the open utterance as final.
        let tail = conv.finish();
        let permit = store.pool().acquire_owned().await.expect("the pool is never closed");
        let done = tokio::task::spawn_blocking(move || {
            let mut pieces = session.accept(&tail);
            pieces.extend(session.finish());
            drop(permit);
            pieces
        })
        .await;
        match done {
            Ok(pieces) => {
                for p in pieces {
                    yield p.into();
                }
            }
            Err(e) => yield ListenEvent::Error { message: format!("decode: {e}") },
        }
    }
}
