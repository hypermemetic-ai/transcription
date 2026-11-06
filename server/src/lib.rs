//! The transcription library served on its own: a Plexus service named
//! `transcription` with `speech` (`listen`) and `models` at its root, behind
//! plexus-login (plexus-starter's login layer, without its page: an agent
//! signs in over the socket, `synapse … _login transcription --key K`).
//!
//! - **Login is always on.** `--auth-provider` names at least one provider
//!   (`key` for agent keys); a caller with no credential reaches only the
//!   `auth` door. Roles are plexus-login's `SimpleAccess`: global, in memory,
//!   the `--admin` principals admin from the start, anyone else waiting
//!   (`auth.waiting`) until an admin grants a role (`auth.grant --role
//!   transcriber --scope '*'`). A restart forgets every grant but the admins'.
//! - **Who may**: `transcriber` may `speech.listen`, `models.list`,
//!   `models.search`; `admin` everything (`models.download`, `models.config`,
//!   `auth.*`).

use std::net::SocketAddr;
use std::sync::Arc;

use plexus_rpc::core::plexus::{Activation, DynamicHub as DynamicService, SessionValidator};
use plexus_login::simple::SimpleAccess;
use plexus_login::sessions::Sessions;
use plexus_login::{Branding, LoginOptions, Wired, Wiring};
use plexus_rpc::transport::websocket::serve_websocket_with;
use plexus_rpc::transport::{ServerHandle, UpgradeOptions, WebSocketConfig};
use transcription::Transcription;

/// The service's name: what synapse calls the backend.
pub const SERVICE: &str = "transcription";

const BRANDING: Branding = Branding { app: SERVICE, version: env!("CARGO_PKG_VERSION"), mark: "t", motto: "" };

/// The login's flags.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct LoginArgs {
    /// Admit callers through this identity provider (repeatable): `key`
    /// (agent keys, as synapse signs in), `idp=<issuer>,<audience>`,
    /// `oidc:<name>=<issuer>,<audience>`, `auth0=…`, `workos=…`, `nostr=…`.
    #[arg(long = "auth-provider", env = "TRANSCRIPTION_AUTH_PROVIDERS", value_name = "SPEC", value_delimiter = ';', required = true)]
    pub auth_provider: Vec<String>,

    /// A principal who is admin from the start (repeatable): `key:<hex>`,
    /// `oidc:<provider>/<sub>`, … — `synapse _key show NAME` prints a key's.
    #[arg(long = "admin", env = "TRANSCRIPTION_ADMINS", value_name = "PRINCIPAL", value_delimiter = ';')]
    pub admin: Vec<String>,

    /// How long a sign-in lasts.
    #[arg(long, env = "TRANSCRIPTION_SESSION_HOURS", default_value = "12", value_name = "HOURS")]
    pub session_hours: i64,
}

/// The service, wired: its activations, `auth`, the gate.
pub struct Login {
    pub service: Arc<DynamicService>,
    pub wired: Wired<SimpleAccess>,
}

impl Login {
    /// Registers `t`'s activations on a service behind the login.
    pub fn wire(args: &LoginArgs, t: &Transcription, ws_port: u16) -> anyhow::Result<Self> {
        anyhow::ensure!(!args.auth_provider.is_empty(), "--auth-provider is required: login is always on");
        anyhow::ensure!(args.session_hours > 0, "--session-hours is at least 1");
        let opts = LoginOptions {
            session_ttl_secs: args.session_hours * 3600,
            // No shell, so no page: agents sign in over the socket.
            page: false,
            ..LoginOptions::new(BRANDING, ws_port, ws_port)
        }
        .with_providers(&args.auth_provider)?;
        let wiring = Wiring::new(opts)?;
        let access = SimpleAccess::new(wiring.sessions(), &args.admin)?;
        let wired = wiring.finish(access.clone(), None)?;
        let service = DynamicService::new(SERVICE).with_auth(access.auth_setup(wired.backend.clone())).register(t.speech()).register(t.models());
        access.seed_roles(&service);
        if args.admin.is_empty() {
            tracing::warn!("no --admin: nobody can grant a role; sign in, read your principal off auth.whoami, restart with --admin <it>");
        }
        Ok(Login { service: Arc::new(service), wired })
    }

    /// The sessions table (a test mints a caller's token here).
    pub fn sessions(&self) -> &Arc<Sessions> {
        &self.wired.sessions
    }
}

/// The socket, serving.
pub struct Serving {
    pub addr: SocketAddr,
    handle: ServerHandle,
}

impl Serving {
    pub async fn stop(self) {
        let _ = self.handle.stop();
        self.handle.stopped().await;
    }

    pub async fn stopped(&self) {
        self.handle.clone().stopped().await;
    }
}

/// Serves `login`'s service on `addr` (port 0: a free one).
pub async fn start(login: &Login, addr: SocketAddr) -> anyhow::Result<Serving> {
    let module = DynamicService::arc_into_rpc_module(login.service.clone()).map_err(|e| anyhow::anyhow!("rpc module: {e}"))?;
    let mut config = WebSocketConfig::new(addr.port());
    config.addr = addr;
    config.allowed_origins = login.wired.origins.clone();
    let mut upgrade = UpgradeOptions::default();
    let mut validator: Option<Arc<dyn SessionValidator>> = None;
    if let Some(gate) = &login.wired.gate {
        validator = Some(gate.clone());
        upgrade.anonymous_view = true;
        upgrade.credential_cookies = vec!["access_token".into()];
        upgrade.protected_resource = Some(login.wired.protected.clone());
    }
    let (handle, addr) = serve_websocket_with(module, config, validator, upgrade).await.map_err(|e| anyhow::anyhow!("cannot serve on {addr}: {e}"))?;
    Ok(Serving { addr, handle })
}

/// The schema hash, for the registry.
pub fn schema(service: &DynamicService) -> plexus_rpc::core::plexus::PluginSchema {
    Activation::plugin_schema(service)
}
