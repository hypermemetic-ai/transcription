use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use clap::Parser;
use plexus_transport::registry::{self, RegistryRegistration};
use transcription::{Catalog, Options, Transcription};
use transcription_server::{schema, start, Login, LoginArgs, SERVICE};

/// transcription: live audio to text with local models, a Plexus service.
#[derive(Parser, Debug)]
#[command(name = "transcription", version)]
struct Args {
    /// The WebSocket's port.
    #[arg(short, long, env = "TRANSCRIPTION_PORT", default_value = "4480")]
    port: u16,

    /// The address it binds: loopback by default.
    #[arg(long, env = "TRANSCRIPTION_BIND", default_value = "127.0.0.1", value_name = "ADDR")]
    bind: IpAddr,

    /// Where models are installed (default: <data-dir>/models).
    #[arg(long, env = "TRANSCRIPTION_MODELS_DIR", value_name = "DIR")]
    models_dir: Option<PathBuf>,

    /// Where the config (transcription.toml) and the models live (default:
    /// $XDG_DATA_HOME/transcription, else ~/.local/share/transcription).
    #[arg(long, env = "TRANSCRIPTION_DATA_DIR", value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// More catalog entries (`[[model]]` tables), or replacements by id.
    #[arg(long, env = "TRANSCRIPTION_CATALOG", value_name = "FILE")]
    catalog: Option<PathBuf>,

    /// Fetch the built-in archives from this base URL instead of GitHub's
    /// sherpa-onnx release.
    #[arg(long, env = "TRANSCRIPTION_MIRROR", value_name = "URL")]
    mirror: Option<String>,

    /// `models search` reads the catalog only, not the GitHub release.
    #[arg(long)]
    offline_search: bool,

    /// Load this model at start, so the first listen does not wait for it.
    #[arg(long, value_name = "ID")]
    warm: Vec<String>,

    #[command(flatten)]
    login: LoginArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();

    let mut opts = Options::default_dirs()?;
    if let Some(d) = &args.data_dir {
        opts.models_dir = d.join("models");
        opts.config_path = Some(d.join("transcription.toml"));
    }
    if let Some(m) = &args.models_dir {
        opts.models_dir = m.clone();
    }
    if let Some(f) = &args.catalog {
        let text = std::fs::read_to_string(f).map_err(|e| anyhow::anyhow!("--catalog {}: {e}", f.display()))?;
        opts.catalog = Catalog::new(Some(&text))?;
    }
    opts.mirror = args.mirror.clone();
    if args.offline_search {
        opts.release_api = None;
    }
    tracing::info!(models = %opts.models_dir.display(), config = ?opts.config_path, "opening");
    let t = Transcription::open(opts)?;
    for id in &args.warm {
        t.warm(id).await.map_err(anyhow::Error::msg)?;
    }

    let login = Login::wire(&args.login, &t, args.port)?;
    let serving = start(&login, SocketAddr::new(args.bind, args.port)).await?;
    tracing::info!("{SERVICE} on ws://{}", serving.addr);

    let s = schema(&login.service);
    let reg = (!registry::registry_disabled_by_env()).then(|| RegistryRegistration {
        url: registry::registry_url_from_env(),
        name: s.namespace.clone(),
        host: "127.0.0.1".into(),
        port: serving.addr.port(),
        version: s.version.clone(),
        schema_hash: s.hash.clone(),
    });
    let registered = reg.clone().map(|reg| tokio::spawn(async move { registry::register_with_retry(&reg).await }));
    tokio::select! {
        _ = serving.stopped() => tracing::info!("the socket stopped"),
        _ = shutdown() => tracing::info!("stopping"),
    }
    if let (Some(reg), Some(registered)) = (reg, registered) {
        if registered.await.unwrap_or(false) {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(3), registry::deregister(&reg)).await;
        }
    }
    serving.stop().await;
    Ok(())
}

async fn shutdown() {
    use tokio::signal::unix::{signal, SignalKind};
    match signal(SignalKind::terminate()) {
        Ok(mut term) => tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        },
        Err(_) => {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}
