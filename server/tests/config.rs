//! `models config`: a file's text, checked (an unknown key refused by name,
//! the old config kept), kept on disk, applied to the next listen.

mod support;

use serde_json::json;
use support::*;

fn config_toml(evs: &[Ev]) -> String {
    evs.iter().filter_map(|e| e.data()).find(|d| d["type"] == "config").map(|d| d["toml"].as_str().unwrap().to_string()).unwrap_or_else(|| panic!("{evs:?}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_key_is_refused_by_name_and_the_old_config_kept() {
    let s = Server::scratch(None).await;
    let admin = s.connect(ADMIN).await;
    let example = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../transcription.toml")).unwrap();
    let set = admin.call("models", "config", json!({ "config": example })).await;
    let kept = config_toml(&set);
    assert!(kept.contains("decode_workers = 8"), "{kept}");
    let on_disk = std::fs::read_to_string(s.scratch_path().join("transcription.toml")).unwrap();
    assert_eq!(on_disk, kept);

    for (bad, named) in [
        ("default_model = \"parakeet-tdt-v2\"\nthreds = 4\n", "threds"),
        ("[vad]\nthreshhold = 0.3\n", "threshhold"),
        ("[models.parakeet-tdt-v2]\npartial = 100\n", "partial"),
        ("default_model = \"whisper-huge\"\n", "whisper-huge"),
        ("threads = 0\n", "threads"),
    ] {
        let evs = admin.call("models", "config", json!({ "config": bad })).await;
        let e = evs.iter().find_map(|e| e.data()).unwrap();
        assert_eq!(e["type"], "error", "{bad}: {e}");
        assert!(e["message"].as_str().unwrap().contains(named), "{bad}: {e}");
        // Nothing changed: the call with nothing shows the old one.
        assert_eq!(config_toml(&admin.call("models", "config", json!({})).await), kept);
        assert_eq!(std::fs::read_to_string(s.scratch_path().join("transcription.toml")).unwrap(), kept);
    }

    // Applied to the next listen: the default model is the one listen names.
    let set = admin.call("models", "config", json!({ "config": "default_model = \"moonshine-tiny-en\"\n" })).await;
    assert!(config_toml(&set).contains("moonshine-tiny-en"));
    s.grant(ALICE, "transcriber").await;
    let evs = s.connect(ALICE).await.listen_all(json!({}), vec![0; 3200], 640).await;
    assert!(evs.iter().find_map(|e| e.data()).unwrap()["message"].as_str().unwrap().contains("moonshine-tiny-en"), "{evs:?}");
    s.stop().await;

    // And read back at the next start.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("transcription.toml"), &kept).unwrap();
    let t = transcription::Transcription::open(transcription::Options {
        models_dir: dir.path().join("models"),
        config_path: Some(dir.path().join("transcription.toml")),
        catalog: transcription::Catalog::new(None).unwrap(),
        mirror: None,
        release_api: None,
    })
    .unwrap();
    assert_eq!(t.config().decode_workers, 8);
}
