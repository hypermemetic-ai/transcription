//! Decode a raw 16 kHz mono s16le file through the engine directly (no
//! socket): every piece, and the real-time factor. A tool for tuning and
//! for the perf numbers.
//!
//!   cargo run --release --example decode -- MODELS_DIR MODEL FILE.s16 [CHUNK_BYTES] [--realtime]

use std::time::{Duration, Instant};

use plexus_transcription::catalog::{Catalog, VAD_ID};
use plexus_transcription::config::Config;
use plexus_transcription::engine::{self, Load, Piece};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (dir, id, file) = (std::path::PathBuf::from(&a[1]), &a[2], &a[3]);
    let chunk: usize = a.get(4).and_then(|c| c.parse().ok()).unwrap_or(640);
    let realtime = a.iter().any(|x| x == "--realtime");
    let cat = Catalog::new(None)?;
    let entry = cat.get(id).ok_or_else(|| anyhow::anyhow!("no model {id}"))?.clone();
    let cfg = Config::default();
    let s = cfg.resolve(id, entry.kind);
    let t0 = Instant::now();
    let eng = engine::load(&Load { entry, dir: dir.join(id), vad: dir.join(VAD_ID).join("silero_vad.onnx"), threads: s.threads })?;
    let load = t0.elapsed();
    let bytes = std::fs::read(file)?;
    let samples: Vec<f32> = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
    let audio_s = samples.len() as f64 / 16_000.0;
    let mut sess = eng.open(&s)?;
    let per = chunk / 2;
    let mut busy = Duration::ZERO;
    let start = Instant::now();
    let mut first_partial = None;
    let show = |p: &Piece, at: f64| match p {
        Piece::Partial { text } => println!("{at:7.2}s partial {text}"),
        Piece::Final { text, start_ms, end_ms } => println!("{at:7.2}s FINAL   [{start_ms}-{end_ms}] {text}"),
    };
    for (i, c) in samples.chunks(per).enumerate() {
        if realtime {
            let due = Duration::from_secs_f64((i * per) as f64 / 16_000.0);
            if let Some(w) = due.checked_sub(start.elapsed()) {
                std::thread::sleep(w);
            }
        }
        let t = Instant::now();
        let ps = sess.accept(c);
        let took = t.elapsed();
        busy += took;
        if took > Duration::from_millis(100) {
            println!("  slow accept #{i} at {:.2}s audio: {took:?}", ((i + 1) * per) as f64 / 16_000.0);
        }
        let at = if realtime { start.elapsed().as_secs_f64() } else { ((i + 1) * per) as f64 / 16_000.0 };
        for p in &ps {
            if first_partial.is_none() && matches!(p, Piece::Partial { .. }) {
                first_partial = Some(at);
            }
            show(p, at);
        }
    }
    let t = Instant::now();
    for p in sess.finish() {
        show(&p, audio_s);
    }
    println!("  finish: {:?}", t.elapsed());
    busy += t.elapsed();
    println!("model={id} threads={} load={load:?} audio={audio_s:.2}s decode={:.3}s rtf={:.3}", s.threads, busy.as_secs_f64(), busy.as_secs_f64() / audio_s);
    Ok(())
}
