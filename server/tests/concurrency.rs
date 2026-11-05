//! Many listeners at once: one model loaded and shared, every call right,
//! memory bounded.

mod support;

use std::time::Instant;

use serde_json::json;
use support::*;

async fn eight(model: &str, bound: f64) {
    need(&[model]);
    let s = Server::over_models().await;
    s.grant(ALICE, "transcriber").await;
    // Warm: the model loaded, one call run, before memory is measured.
    let c = s.connect(ALICE).await;
    let _ = c.listen_all(json!({ "model": model }), walk(), 4096).await;
    let before = rss_mib();
    let started = Instant::now();
    let calls = (0..8).map(|_| {
        let s = &s;
        async move {
            let c = s.connect(ALICE).await;
            c.listen_all(json!({ "model": model }), walk(), 640).await
        }
    });
    let all = futures::future::join_all(calls).await;
    let wall = started.elapsed();
    let after = rss_mib();
    for (i, evs) in all.iter().enumerate() {
        let w = wer(&walk_text(), &finals(evs).join(" "));
        assert!(w <= bound, "{model} listener {i}: WER {w}: {evs:?}");
        assert_eq!(finals(evs).len(), 3, "{model} listener {i}: {evs:?}");
    }
    let per = (after - before).max(0.0) / 8.0;
    println!("{model}: 8 × 13.9 s in {wall:.2?}; RSS {before:.0} → {after:.0} MiB ({per:.1} MiB a listener)");
    // One model shared: eight listeners cost their streams and buffers
    // (under 100 MiB each), never another copy of the model (850+ MiB).
    assert!(per < 100.0, "{model}: RSS grew {:.0} MiB for 8 listeners", after - before);
    s.stop().await;
}

/// One test, the models in turn: the process's RSS is what is measured, so
/// nothing else may load a model meanwhile.
#[tokio::test(flavor = "multi_thread")]
async fn eight_listeners_all_finish_right_streaming_and_offline() {
    eight("nemotron-en-560ms", 0.15).await;
    eight("parakeet-tdt-v2", 0.10).await;
}
