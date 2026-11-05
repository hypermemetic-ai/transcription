//! Who may: `transcriber` listens and reads the models; `admin` everything;
//! a caller with no role is refused by the gate, typed.

mod support;

use serde_json::json;
use support::*;

fn forbidden(evs: &[Ev]) -> bool {
    evs.iter().any(|e| matches!(e, Ev::Failed(_)) && kind(e).contains("forbidden"))
}

#[tokio::test(flavor = "multi_thread")]
async fn no_role_is_refused_and_transcriber_is_served() {
    let s = Server::scratch(None).await;
    let bob = s.connect(BOB).await;
    // No role: every method of the service is the gate's no, typed.
    let evs = bob.listen_all(json!({}), vec![0; 6400], 640).await;
    assert!(forbidden(&evs), "listen, no role: {evs:?}");
    assert!(forbidden(&bob.call("models", "list", json!({})).await));
    assert!(forbidden(&bob.call("models", "search", json!({ "query": "parakeet" })).await));

    // transcriber: listen (here refused in band — nothing is downloaded —
    // which is the method running), list, search.
    s.grant(ALICE, "transcriber").await;
    let alice = s.connect(ALICE).await;
    let evs = alice.listen_all(json!({ "model": "parakeet-tdt-v2" }), vec![0; 6400], 640).await;
    assert!(!forbidden(&evs), "{evs:?}");
    assert!(evs.iter().any(|e| e.data().is_some_and(|d| d["message"].as_str().unwrap_or("").contains("models download"))), "{evs:?}");
    let list = alice.call("models", "list", json!({})).await;
    assert!(list.iter().filter_map(|e| e.data()).any(|d| d["id"] == "nemotron-en-560ms" && d["default"] == true), "{list:?}");
    let found = alice.call("models", "search", json!({ "query": "parakeet" })).await;
    assert!(found.iter().filter_map(|e| e.data()).any(|d| d["id"] == "parakeet-tdt-v2"), "{found:?}");
    // …but not download or config: those are admin's.
    assert!(forbidden(&alice.call("models", "download", json!({ "id": "parakeet-tdt-v2" })).await));
    assert!(forbidden(&alice.call("models", "config", json!({})).await));

    // admin holds every grant.
    let admin = s.connect(ADMIN).await;
    let cfg = admin.call("models", "config", json!({})).await;
    assert!(cfg.iter().filter_map(|e| e.data()).any(|d| d["type"] == "config"), "{cfg:?}");
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_roles_are_declared_for_the_gate() {
    let s = Server::scratch(None).await;
    let roles = s.login.service.declared_roles();
    let mut t = roles.get("transcriber").cloned().unwrap_or_default();
    t.sort();
    assert_eq!(t, ["models.list", "models.search", "speech.listen"], "{roles:?}");
    s.stop().await;
}

/// A host service registers the one activation beside its own: `listen` and
/// its `models` child under `transcription`, the roles qualified by path.
#[tokio::test(flavor = "multi_thread")]
async fn a_host_registers_the_activation_with_its_roles() {
    use futures::StreamExt;
    use plexus_core::plexus::DynamicHub as DynamicService;
    let dir = tempfile::tempdir().unwrap();
    let t = transcription::Transcription::open(transcription::Options {
        models_dir: dir.path().join("models"),
        config_path: None,
        catalog: transcription::Catalog::new(None).unwrap(),
        mirror: None,
        release_api: None,
    })
    .unwrap();
    let host = DynamicService::new("myapp").register(transcription::TranscriptionActivation::new(t));
    let mut roles = host.declared_roles().remove(transcription::ROLE).unwrap();
    roles.sort();
    assert_eq!(roles, ["transcription.listen", "transcription.models.list", "transcription.models.search"]);
    let items: Vec<_> = host.route("transcription.models.list", serde_json::json!({}), None).await.unwrap().collect().await;
    let text = serde_json::to_string(&items).unwrap();
    assert!(text.contains("nemotron-en-560ms") && text.contains("parakeet-tdt-v2"), "{text}");
}
