//! `speech.listen` over a real socket: PCM in as binary frames, text out on
//! the same call. Needs nemotron-en-560ms and parakeet-tdt-v2 installed
//! (support::models_dir).

mod support;

use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use support::*;

/// The words of the synthetic clip each model must get within: Nemotron
/// hears espeak's "kiln" as "kill" and drops the first "So" (2 of 23).
const WER_STREAMING: f64 = 0.15;
const WER_OFFLINE: f64 = 0.10;

async fn transcribe(model: &str, bound: f64) {
    need(&[model]);
    let s = Server::over_models().await;
    s.grant(ALICE, "transcriber").await;
    let c = s.connect(ALICE).await;
    let evs = c.listen_all(json!({ "model": model }), walk(), 640).await;
    let finals = finals(&evs);
    let text = finals.join(" ");
    let w = wer(&walk_text(), &text);
    println!("{model}: WER {w:.3} — {finals:?}");
    assert!(w <= bound, "{model}: WER {w:.3} > {bound}: {text:?}\n{evs:?}");
    // VAD-segmented: one final per utterance, each with its place.
    assert_eq!(finals.len(), 3, "{evs:?}");
    for d in evs.iter().filter_map(|e| e.data()).filter(|d| d["type"] == "final") {
        assert!(d["start_ms"].as_u64().unwrap() < d["end_ms"].as_u64().unwrap(), "{d}");
    }
    assert!(evs.iter().any(|e| e.data().is_some_and(|d| d["type"] == "partial")), "partials: {evs:?}");
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streaming_model_transcribes_the_clip() {
    transcribe("nemotron-en-560ms", WER_STREAMING).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_offline_model_transcribes_the_clip() {
    transcribe("parakeet-tdt-v2", WER_OFFLINE).await;
}

/// The input stays open until a partial has come back: text arrives while
/// audio is still streaming in, on the same call.
#[tokio::test(flavor = "multi_thread")]
async fn partials_arrive_before_the_input_ends() {
    for model in ["nemotron-en-560ms", "parakeet-tdt-v2"] {
        need(&[model]);
        let s = Server::over_models().await;
        s.grant(ALICE, "transcriber").await;
        let c = s.connect(ALICE).await;
        let (tx, rx) = futures::channel::mpsc::unbounded::<Bytes>();
        // The first utterance alone (3.7 s), and the input left open.
        for chunk in walk()[..(3_700 * 32)].chunks(640) {
            tx.unbounded_send(Bytes::copy_from_slice(chunk)).unwrap();
        }
        let mut events = Box::pin(c.listen(json!({ "model": model }), rx));
        let first = tokio::time::timeout(Duration::from_secs(60), async {
            while let Some(e) = events.next().await {
                if e.data().is_some_and(|d| d["type"] == "partial") {
                    return e;
                }
                assert!(e.data().is_some_and(|d| d["type"] != "error"), "{e:?}");
            }
            panic!("the call ended with no partial");
        })
        .await
        .expect("a partial while the input is open");
        println!("{model}: first partial {first:?}, input still open");
        // Now the rest, then the end: the call says the finals and is done.
        for chunk in walk()[(3_700 * 32)..].chunks(640) {
            tx.unbounded_send(Bytes::copy_from_slice(chunk)).unwrap();
        }
        drop(tx);
        let rest: Vec<Ev> = events.collect().await;
        assert!(!finals(&rest).is_empty(), "{rest:?}");
        s.stop().await;
    }
}

/// 48 kHz stereo in, said as such: resampled and downmixed, same words.
#[tokio::test(flavor = "multi_thread")]
async fn another_rate_and_channels_are_converted() {
    need(&["parakeet-tdt-v2"]);
    let s = Server::over_models().await;
    s.grant(ALICE, "transcriber").await;
    let c = s.connect(ALICE).await;
    // Each 16 kHz sample three times (a crude upsample), on two channels.
    let pcm = walk();
    let mut st = Vec::with_capacity(pcm.len() * 6);
    for smp in pcm.chunks_exact(2) {
        for _ in 0..6 {
            st.extend_from_slice(smp);
        }
    }
    let evs = c.listen_all(json!({ "model": "parakeet-tdt-v2", "rate": 48000, "channels": 2, "format": "s16le" }), st, 4096).await;
    let w = wer(&walk_text(), &finals(&evs).join(" "));
    assert!(w <= WER_OFFLINE, "WER {w}: {evs:?}");
    // f32le, mono, 16 kHz.
    let f: Vec<u8> = pcm.chunks_exact(2).flat_map(|b| (i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).to_le_bytes()).collect();
    let evs = c.listen_all(json!({ "model": "parakeet-tdt-v2", "format": "f32le" }), f, 4096).await;
    let w = wer(&walk_text(), &finals(&evs).join(" "));
    assert!(w <= WER_OFFLINE, "f32le WER {w}: {evs:?}");
    s.stop().await;
}

/// A model not downloaded, or none such: refused in band, naming what to run.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_not_downloaded_is_refused_in_band() {
    let s = Server::scratch(None).await;
    s.grant(ALICE, "transcriber").await;
    let c = s.connect(ALICE).await;
    let evs = c.listen_all(json!({ "model": "parakeet-tdt-v3" }), vec![0; 3200], 640).await;
    let e = evs.iter().find_map(|e| e.data()).expect("an event");
    assert_eq!(e["type"], "error");
    assert!(e["message"].as_str().unwrap().contains("models download --id parakeet-tdt-v3"), "{e}");
    let evs = c.listen_all(json!({ "model": "nope" }), vec![0; 3200], 640).await;
    assert!(evs.iter().find_map(|e| e.data()).unwrap()["message"].as_str().unwrap().contains("no model `nope`"));
    // No --model: the configured default, named.
    let evs = c.listen_all(json!({}), vec![0; 3200], 640).await;
    assert!(evs.iter().find_map(|e| e.data()).unwrap()["message"].as_str().unwrap().contains("nemotron-en-560ms"), "{evs:?}");
    s.stop().await;
}
