//! The perf row: per model, load time and memory, the real-time factor of
//! one listener, first-partial and end-of-utterance → final latency at live
//! pace, and how many live listeners one model keeps up with. Not part of
//! the gate (it takes minutes); run it on demand:
//!
//!   cargo test --release -p transcription-server --test perf -- --ignored --nocapture
//!
//! `TRANSCRIPTION_PERF_MODELS=a,b` picks the models; `TRANSCRIPTION_PERF_N=1,8,16`
//! the listener counts (at most 60: a principal holds 64 sockets);
//! `TRANSCRIPTION_PERF_CONFIG` a config's TOML to run under.

mod support;

use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use support::*;

/// Where the clip's speech begins (its first VAD segment).
const SPEECH_MS: f64 = 74.0;

struct Live {
    first_partial_ms: Option<f64>,
    final_lat_ms: Vec<f64>,
    /// From the last byte sent to the call's end.
    drain_ms: f64,
    wer: f64,
}

/// One listener at live pace: 20 ms frames on a clock.
async fn live(c: &Client, model: &str) -> Live {
    let pcm = walk();
    let (tx, rx) = futures::channel::mpsc::unbounded::<Bytes>();
    let t0 = Instant::now();
    let sender = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
        for ch in pcm.chunks(640) {
            tick.tick().await;
            if tx.unbounded_send(Bytes::copy_from_slice(ch)).is_err() {
                break;
            }
        }
        Instant::now()
    });
    let mut events = Box::pin(c.listen(json!({ "model": model }), rx));
    let (mut first, mut lat, mut texts) = (None, Vec::new(), Vec::new());
    while let Some(e) = events.next().await {
        let at = t0.elapsed().as_secs_f64() * 1000.0;
        let Some(d) = e.data() else { panic!("{e:?}") };
        match d["type"].as_str().unwrap() {
            "partial" if first.is_none() => first = Some(at - SPEECH_MS),
            "final" => {
                lat.push(at - d["end_ms"].as_f64().unwrap());
                texts.push(d["text"].as_str().unwrap().to_string());
            }
            "error" => panic!("{d}"),
            _ => {}
        }
    }
    let sent = sender.await.unwrap();
    Live { first_partial_ms: first, final_lat_ms: lat, drain_ms: sent.elapsed().as_secs_f64() * 1000.0, wer: wer(&walk_text(), &texts.join(" ")) }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0, f64::max)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn perf() {
    let models: Vec<String> = std::env::var("TRANSCRIPTION_PERF_MODELS")
        .map(|m| m.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| ["nemotron-en-560ms", "zipformer-en-20m", "parakeet-tdt-v2", "moonshine-tiny-en"].map(String::from).to_vec());
    let ns: Vec<usize> = std::env::var("TRANSCRIPTION_PERF_N").map(|n| n.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_else(|_| vec![1, 8, 16, 24, 32, 48]);
    let audio_s = walk().len() as f64 / 32_000.0;
    let s = Server::over_models().await;
    s.grant(ALICE, "transcriber").await;
    let c = s.connect(ALICE).await;
    if let Ok(text) = std::env::var("TRANSCRIPTION_PERF_CONFIG") {
        let set = s.connect(ADMIN).await.call("models", "config", json!({ "config": text })).await;
        assert!(set.iter().any(|e| e.data().is_some_and(|d| d["type"] == "config")), "{set:?}");
    }
    println!("config: threads {} per decode, decode_workers {}", s.t.config().threads, s.t.config().decode_workers);
    for model in &models {
        need(&[model]);
        let rss0 = rss_mib();
        let t = Instant::now();
        s.t.warm(model).await.unwrap();
        let load = t.elapsed();
        let rss_model = rss_mib() - rss0;
        // One listener, fed as fast as it takes it: decode time ÷ audio time.
        let t = Instant::now();
        let evs = c.listen_all(json!({ "model": model }), walk(), 4096).await;
        let rtf = t.elapsed().as_secs_f64() / audio_s;
        let w = wer(&walk_text(), &finals(&evs).join(" "));
        println!("\n== {model}: load {load:.2?}, +{rss_model:.0} MiB loaded; one listener flat out: RTF {rtf:.3}, WER {w:.3}");
        println!("   N | first partial ms (mean/max) | end→final ms (mean/max) | drain ms (max) | WER max | RSS MiB");
        // One socket per listener, made once (the per-IP quota counts them).
        let mut conns = Vec::new();
        for _ in 0..*ns.iter().max().unwrap() {
            conns.push(s.connect(ALICE).await);
        }
        for &n in &ns {
            let rss_before = rss_mib();
            let runs = futures::future::join_all(conns[..n].iter().map(|c| live(c, model))).await;
            let fp: Vec<f64> = runs.iter().filter_map(|r| r.first_partial_ms).collect();
            let fl: Vec<f64> = runs.iter().flat_map(|r| r.final_lat_ms.clone()).collect();
            let dr: Vec<f64> = runs.iter().map(|r| r.drain_ms).collect();
            let wm = runs.iter().map(|r| r.wer).fold(0.0, f64::max);
            println!(
                "{n:4} | {:8.0} / {:8.0}          | {:7.0} / {:7.0}         | {:8.0}       | {wm:.3}   | {:.0} (+{:.0})",
                mean(&fp),
                max(&fp),
                mean(&fl),
                max(&fl),
                max(&dr),
                rss_mib(),
                rss_mib() - rss_before
            );
            // Fallen behind live: stop raising N for this model.
            if max(&fl) > 5_000.0 {
                println!("     past {n} listeners the finals lag more than 5 s behind the speech: not live");
                break;
            }
        }
    }
    s.stop().await;
}
