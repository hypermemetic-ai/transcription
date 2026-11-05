//! `models download` against a local HTTP fixture: progress, the size and
//! sha256 checked, extracted, installed; a cut download resumed with
//! `Range`; a second download said to be installed already; a bad archive
//! refused and removed.

mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use support::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A model archive the way the release has them: one directory, the
/// engine's files in it (not loadable — a download only checks they are
/// there).
fn archive() -> Vec<u8> {
    let mut tar = tar::Builder::new(Vec::new());
    for (name, body) in [("encoder-epoch-1.int8.onnx", vec![1u8; 300_000]), ("decoder-epoch-1.onnx", vec![2; 1000]), ("joiner-epoch-1.onnx", vec![3; 1000]), ("tokens.txt", b"<blk> 0\n".to_vec())] {
        let mut h = tar::Header::new_gnu();
        h.set_size(body.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        tar.append_data(&mut h, format!("sherpa-onnx-fixture/{name}"), &body[..]).unwrap();
    }
    let raw = tar.into_inner().unwrap();
    let mut out = Vec::new();
    let mut bz = bzip2::read::BzEncoder::new(&raw[..], bzip2::Compression::fast());
    std::io::Read::read_to_end(&mut bz, &mut out).unwrap();
    out
}

struct Fixture {
    url: String,
    /// GETs of the model archive.
    model_gets: Arc<AtomicUsize>,
    /// The Range headers asked.
    ranges: Arc<std::sync::Mutex<Vec<String>>>,
}

/// Serves `/model.tar.bz2` (the first GET cut halfway) and `/vad.onnx`.
async fn serve(model: Vec<u8>, vad: Vec<u8>) -> Fixture {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let model_gets = Arc::new(AtomicUsize::new(0));
    let ranges = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (m, r) = (model_gets.clone(), ranges.clone());
    let (model, vad) = (Arc::new(model), Arc::new(vad));
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = l.accept().await else { return };
            let (m, r, model, vad) = (m.clone(), r.clone(), model.clone(), vad.clone());
            tokio::spawn(async move {
                let mut req = Vec::new();
                let mut buf = [0u8; 1024];
                while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = sock.read(&mut buf).await.unwrap();
                    if n == 0 {
                        return;
                    }
                    req.extend_from_slice(&buf[..n]);
                }
                let req = String::from_utf8_lossy(&req).to_string();
                let path = req.split_whitespace().nth(1).unwrap_or("").to_string();
                let range = req.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("range: bytes=").map(|v| v.trim().to_string()));
                let body: &[u8] = match path.as_str() {
                    "/model.tar.bz2" => &model,
                    "/vad.onnx" => &vad,
                    _ => {
                        let _ = sock.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").await;
                        return;
                    }
                };
                let first_model_get = path == "/model.tar.bz2" && m.fetch_add(1, Ordering::SeqCst) == 0;
                if let Some(range) = &range {
                    r.lock().unwrap().push(range.clone());
                    let from: usize = range.trim_end_matches('-').parse().unwrap();
                    let rest = &body[from..];
                    let head = format!("HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {from}-{}/{}\r\nconnection: close\r\n\r\n", rest.len(), body.len() - 1, body.len());
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(rest).await;
                    return;
                }
                let head = format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len());
                let _ = sock.write_all(head.as_bytes()).await;
                if first_model_get {
                    // Cut: half the body, then the connection goes.
                    let _ = sock.write_all(&body[..body.len() / 2]).await;
                    let _ = sock.flush().await;
                    return;
                }
                let _ = sock.write_all(body).await;
            });
        }
    });
    Fixture { url, model_gets, ranges }
}

fn sha(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn catalog(url: &str, model: &[u8], model_sha: &str, vad: &[u8]) -> String {
    format!(
        r#"[[model]]
id = "fixture"
description = "a fixture model"
kind = "streaming"
family = "online_transducer"
languages = ["en"]
archive = "fixture.tar.bz2"
url = "{url}/model.tar.bz2"
size = {}
sha256 = "{model_sha}"

[[model]]
id = "silero-vad"
description = "a fixture VAD"
kind = "vad"
family = "silero_vad"
languages = []
archive = "silero_vad.onnx"
url = "{url}/vad.onnx"
size = {}
sha256 = "{}"
"#,
        model.len(),
        vad.len(),
        sha(vad)
    )
}

fn steps(evs: &[Ev]) -> Vec<Value> {
    evs.iter().map(|e| e.data().cloned().unwrap_or_else(|| panic!("{e:?}"))).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_download_is_verified_extracted_installed_and_resumed_after_a_cut() {
    let model = archive();
    let vad = vec![7u8; 50_000];
    let fx = serve(model.clone(), vad.clone()).await;
    let s = Server::scratch(Some(&catalog(&fx.url, &model, &sha(&model), &vad))).await;
    let admin = s.connect(ADMIN).await;

    // 1: the VAD comes first and installs; the model is cut halfway.
    let first = steps(&admin.call("models", "download", json!({ "id": "fixture" })).await);
    println!("{first:#?}");
    assert!(first.iter().any(|d| d["type"] == "installed" && d["id"] == "silero-vad"));
    assert!(first.iter().any(|d| d["type"] == "progress" && d["id"] == "fixture" && d["total"] == model.len()));
    let last = first.last().unwrap();
    assert_eq!(last["type"], "error");
    assert!(last["message"].as_str().unwrap().contains("run it again to resume"), "{last}");
    let part = s.scratch_path().join("models/.partial/fixture.tar.bz2");
    let kept = std::fs::metadata(&part).unwrap().len();
    assert!(kept > 0 && kept < model.len() as u64, "{kept}");
    assert!(!s.scratch_path().join("models/fixture").exists());

    // 2: resumed from where it stopped, verified, extracted, installed.
    let second = steps(&admin.call("models", "download", json!({ "id": "fixture" })).await);
    println!("{second:#?}");
    assert_eq!(second.iter().find(|d| d["type"] == "resumed").unwrap()["from"], kept);
    assert_eq!(fx.ranges.lock().unwrap().as_slice(), [format!("{kept}-")]);
    let v = second.iter().find(|d| d["type"] == "verified").unwrap();
    assert_eq!((v["sha256"].as_str().unwrap(), v["sha256_checked"].as_bool().unwrap()), (sha(&model).as_str(), true));
    let done = second.last().unwrap();
    assert_eq!(done["type"], "installed", "{done}");
    let dir = s.scratch_path().join("models/fixture");
    assert_eq!(done["path"], dir.display().to_string());
    assert!(dir.join(".installed").is_file() && dir.join("tokens.txt").is_file() && dir.join("encoder-epoch-1.int8.onnx").is_file());
    assert!(!part.exists(), "the partial file is gone once installed");
    assert_eq!(fx.model_gets.load(Ordering::SeqCst), 2);

    // 3: installed already: said, nothing fetched.
    let third = steps(&admin.call("models", "download", json!({ "id": "fixture" })).await);
    assert_eq!(third.len(), 1, "{third:?}");
    assert_eq!(third[0]["type"], "already_installed");
    assert_eq!(fx.model_gets.load(Ordering::SeqCst), 2);
    let list = admin.call("models", "list", json!({})).await;
    assert!(list.iter().filter_map(|e| e.data()).any(|d| d["id"] == "fixture" && d["installed"] == true));
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_archive_that_does_not_check_is_refused_and_removed() {
    let model = archive();
    let vad = vec![7u8; 50_000];
    let fx = serve(model.clone(), vad.clone()).await;
    let wrong = "0".repeat(64);
    let s = Server::scratch(Some(&catalog(&fx.url, &model, &wrong, &vad))).await;
    let admin = s.connect(ADMIN).await;
    let _cut = admin.call("models", "download", json!({ "id": "fixture" })).await;
    let evs = steps(&admin.call("models", "download", json!({ "id": "fixture" })).await);
    let last = evs.last().unwrap();
    assert_eq!(last["type"], "error");
    assert!(last["message"].as_str().unwrap().contains("sha256"), "{last}");
    assert!(!s.scratch_path().join("models/.partial/fixture.tar.bz2").exists());
    assert!(!s.scratch_path().join("models/fixture").exists());
    // No such id: refused in band.
    let evs = steps(&admin.call("models", "download", json!({ "id": "nope" })).await);
    assert!(evs[0]["message"].as_str().unwrap().contains("no model `nope`"));
    s.stop().await;
}
