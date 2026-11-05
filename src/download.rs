//! `models download`: fetch an archive over HTTP (resuming a cut one with
//! `Range`), check its size and sha256, extract it, check the engine's files
//! are there, and only then mark it installed.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt};
use sha2::{Digest, Sha256};

use crate::catalog::{Entry, Family};
use crate::store::{write_atomic, Store};

/// What a download says on the way.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Bytes of the archive on disk, of its size.
    Progress { bytes: u64, total: u64 },
    /// Where a cut download picks up.
    Resumed { from: u64 },
    /// The archive's size and sha256 hold.
    Verified { sha256: String, checked: bool },
    /// Installed now.
    Installed { path: PathBuf },
    /// Installed before; nothing fetched.
    AlreadyInstalled { path: PathBuf },
}

/// Downloads `entry` into `store`'s models directory: each step, then the
/// end, or why it stopped (a cut download keeps its bytes for the next try).
pub fn download(store: Arc<Store>, entry: Entry) -> impl Stream<Item = Result<Step, String>> + Send + 'static {
    async_stream::stream! {
        if let Some(path) = store.installed(&entry.id) {
            yield Ok(Step::AlreadyInstalled { path });
            return;
        }
        // One download of a model at a time.
        let claimed = store.downloading.lock().unwrap_or_else(|p| p.into_inner()).insert(entry.id.clone());
        if !claimed {
            yield Err(format!("a download of `{}` is under way already", entry.id));
            return;
        }
        let _done = Busy(store.clone(), entry.id.clone());
        let part = store.partial_file(&entry);
        let url = entry.url(store.opts.mirror.as_deref());
        let mut steps = Box::pin(fetch(store.http.clone(), url, part.clone(), entry.size));
        while let Some(step) = steps.next().await {
            let failed = step.is_err();
            yield step;
            if failed {
                return;
            }
        }
        let (p, want) = (part.clone(), entry.sha256.clone());
        let size = entry.size;
        let verified = tokio::task::spawn_blocking(move || verify(&p, size, want.as_deref())).await.map_err(|e| e.to_string()).and_then(|r| r);
        match verified {
            Ok(step) => yield Ok(step),
            Err(e) => {
                // A whole archive that does not check is not resumable.
                let _ = std::fs::remove_file(&part);
                yield Err(e);
                return;
            }
        }
        let (s, e2) = (store.clone(), entry.clone());
        match tokio::task::spawn_blocking(move || install(&s, &e2, &part)).await.map_err(|e| e.to_string()).and_then(|r| r) {
            Ok(path) => yield Ok(Step::Installed { path }),
            Err(e) => yield Err(e),
        }
    }
}

struct Busy(Arc<Store>, String);

impl Drop for Busy {
    fn drop(&mut self) {
        self.0.downloading.lock().unwrap_or_else(|p| p.into_inner()).remove(&self.1);
    }
}

/// Fetches `url` into `part`, from where a cut one stopped.
fn fetch(http: reqwest::Client, url: String, part: PathBuf, total: u64) -> impl Stream<Item = Result<Step, String>> + Send + 'static {
    async_stream::stream! {
        if let Some(dir) = part.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                yield Err(format!("{}: {e}", dir.display()));
                return;
            }
        }
        let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have > total {
            let _ = std::fs::remove_file(&part);
            have = 0;
        }
        if have == total {
            yield Ok(Step::Progress { bytes: have, total });
            return;
        }
        let mut req = http.get(&url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                yield Err(format!("{url}: {e}"));
                return;
            }
        };
        let status = resp.status();
        let append = match status.as_u16() {
            206 if have > 0 => true,
            200 => false,
            416 if have > 0 => {
                // Asked past the end: what is here is all of it, or wrong — verify says which.
                yield Ok(Step::Progress { bytes: have, total });
                return;
            }
            _ => {
                yield Err(format!("{url}: HTTP {status}"));
                return;
            }
        };
        if append {
            yield Ok(Step::Resumed { from: have });
        } else {
            have = 0;
        }
        let file = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&part);
        let mut file = match file {
            Ok(f) => std::io::BufWriter::with_capacity(1 << 20, f),
            Err(e) => {
                yield Err(format!("{}: {e}", part.display()));
                return;
            }
        };
        yield Ok(Step::Progress { bytes: have, total });
        let mut body = resp.bytes_stream();
        let mut said = Instant::now();
        loop {
            let chunk = match tokio::time::timeout(Duration::from_secs(60), body.next()).await {
                Ok(Some(Ok(c))) => c,
                Ok(None) => break,
                Ok(Some(Err(e))) => {
                    let _ = file.flush();
                    yield Err(format!("the download was cut at {have} of {total} bytes ({e}): run it again to resume"));
                    return;
                }
                Err(_) => {
                    let _ = file.flush();
                    yield Err(format!("the download stalled at {have} of {total} bytes: run it again to resume"));
                    return;
                }
            };
            if let Err(e) = file.write_all(&chunk) {
                yield Err(format!("{}: {e}", part.display()));
                return;
            }
            have += chunk.len() as u64;
            if said.elapsed() >= Duration::from_millis(250) {
                said = Instant::now();
                yield Ok(Step::Progress { bytes: have, total });
            }
        }
        if let Err(e) = file.flush().and_then(|_| file.get_ref().sync_all()) {
            yield Err(format!("{}: {e}", part.display()));
            return;
        }
        yield Ok(Step::Progress { bytes: have, total });
        if have < total {
            yield Err(format!("the download was cut at {have} of {total} bytes: run it again to resume"));
        }
    }
}

fn verify(part: &Path, size: u64, sha256: Option<&str>) -> Result<Step, String> {
    let len = std::fs::metadata(part).map_err(|e| format!("{}: {e}", part.display()))?.len();
    if len != size {
        return Err(format!("the archive is {len} bytes, not {size}: it was removed; download it again"));
    }
    let mut f = std::fs::File::open(part).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| e.to_string())?;
    let got = hex::encode(h.finalize());
    if let Some(want) = sha256 {
        if !got.eq_ignore_ascii_case(want) {
            return Err(format!("the archive's sha256 is {got}, not {want}: it was removed; download it again"));
        }
    }
    Ok(Step::Verified { sha256: got, checked: sha256.is_some() })
}

/// Extracts (or places) the verified archive and marks it installed.
fn install(store: &Store, e: &Entry, part: &Path) -> Result<PathBuf, String> {
    let models = &store.opts.models_dir;
    let dest = store.dir_of(&e.id);
    let staging = models.join(format!(".staging-{}", e.id));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|x| format!("{}: {x}", staging.display()))?;
    let root = if e.family == Family::SileroVad || e.archive.ends_with(".onnx") {
        std::fs::copy(part, staging.join("silero_vad.onnx")).map_err(|x| x.to_string())?;
        staging.clone()
    } else {
        let f = std::fs::File::open(part).map_err(|x| x.to_string())?;
        let mut tar = tar::Archive::new(bzip2::read::BzDecoder::new(std::io::BufReader::new(f)));
        tar.unpack(&staging).map_err(|x| format!("extracting {}: {x}", e.archive))?;
        // The archive's one directory, else its files where they are.
        let dirs: Vec<PathBuf> = std::fs::read_dir(&staging).map_err(|x| x.to_string())?.flatten().map(|d| d.path()).collect();
        match dirs.as_slice() {
            [one] if one.is_dir() => one.clone(),
            _ => staging.clone(),
        }
    };
    // The test clips the release ships are not needed to run.
    let _ = std::fs::remove_dir_all(root.join("test_wavs"));
    crate::engine::sherpa::check_files(e.family, &root).map_err(|x| format!("{}: {x:#}", e.id))?;
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::rename(&root, &dest).map_err(|x| format!("{}: {x}", dest.display()))?;
    let _ = std::fs::remove_dir_all(&staging);
    let mark = serde_json::json!({ "id": e.id, "archive": e.archive, "size": e.size, "sha256": e.sha256 });
    write_atomic(&dest.join(".installed"), mark.to_string().as_bytes()).map_err(|x| x.to_string())?;
    let _ = std::fs::remove_file(part);
    Ok(dest)
}
