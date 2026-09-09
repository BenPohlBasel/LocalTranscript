// LocalTranscript-Shell (enrich-Muster): die Shell ist dumm — sie spawnt
// das Python-Backend auf 127.0.0.1:5628, wartet auf /api/health und
// beendet beim Quit NUR, was sie selbst gestartet hat. Ein fremd
// gestartetes gesundes Backend (Terminal-Dev) wird benutzt, nie angefasst.
//
// MERKDATEI (aus enrich nachgezogen, 2026-09-09): app+pid+port landen in
// ~/Library/Application Support/LocalTranscript/app-backend.json. Stürzt
// die App ab, findet der nächste Start sein verwaistes Backend wieder und
// übernimmt es — vorher prüft die ps-Kommandozeile, ob die PID überhaupt
// noch zu einem LocalTranscript-Backend gehört (PIDs werden vom System
// WIEDERVERWENDET; ohne die Probe könnte die App einen wildfremden
// Prozess „übernehmen" und beim Quit beenden).
//
// Die Datei merkt sich AUCH die PID der App, der das Backend gehört.
// Ohne sie hätte eine ZWEITE Instanz das Backend der ersten als eigene
// Waise übernommen und beim Beenden mitgerissen (live aufgetreten
// 2026-09-09: Dev-Build neben installierter App). Übernommen wird nur,
// was einem TOTEN Lauf gehört — läuft die Besitzerin noch, ist das
// Backend fremd und die Merkdatei ihres.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};

/// DER LocalTranscript-Port: 5628 = „LOCT" auf der Telefontastatur
/// (enrich 36742 = „ENRIC", Zotero-Tradition). Vier Buchstaben, nicht
/// fünf: „LOCTR" wäre 56287 und läge damit im EPHEMEREN Bereich
/// (macOS verteilt 49152–65535 selbst) — als fester Dienst-Port
/// untauglich. 5628 liegt im User-Bereich 1024–49151, IANA-unvergeben.
/// Override: LT_SERVE_PORT (eine Quelle je Sprache, s. config.py).
const PORT_STANDARD: u16 = 5628;

fn port() -> u16 {
    static P: std::sync::OnceLock<u16> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("LT_SERVE_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|p| *p >= 1024)
            .unwrap_or(PORT_STANDARD)
    })
}

struct EigenesBackend(Mutex<Option<u32>>);

// ---------- Merkdatei ----------

fn merkdatei() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join("Library/Application Support/LocalTranscript/app-backend.json")
}

fn merk_schreiben(pid: u32) {
    let f = merkdatei();
    if let Some(d) = f.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(
        &f,
        format!(
            "{{\"app\": {}, \"pid\": {pid}, \"port\": {}}}\n",
            std::process::id(),
            port()
        ),
    );
}

fn merk_loeschen() {
    let _ = std::fs::remove_file(merkdatei());
}

fn pid_lebt(pid: u32) -> bool {
    std::process::Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// app+pid+port aus der Merkdatei — bewusst von Hand geparst, die Shell
/// zieht für drei Zahlen keine JSON-Abhängigkeit.
fn merk_lesen() -> Option<(u32, u32, u16)> {
    let roh = std::fs::read_to_string(merkdatei()).ok()?;
    let zahl = |feld: &str| -> Option<u64> {
        let ab = roh.find(feld)? + feld.len();
        roh[ab..]
            .trim_start_matches(|c: char| c == '"' || c == ':' || c.is_whitespace())
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok()
    };
    Some((
        zahl("\"app\"")? as u32,
        zahl("\"pid\"")? as u32,
        zahl("\"port\"")? as u16,
    ))
}

/// Gehört die Merkdatei DIESEM Lauf? Nur dann darf er sie räumen.
fn merk_ist_meine() -> bool {
    merk_lesen().map(|(app, _, _)| app == std::process::id()).unwrap_or(false)
}

fn health_antwortet(frist_s: u64) -> bool {
    let frist = Instant::now() + Duration::from_secs(frist_s);
    loop {
        if let Ok(mut s) = TcpStream::connect_timeout(
            &format!("127.0.0.1:{}", port()).parse().unwrap(),
            Duration::from_millis(500),
        ) {
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let _ = s.write_all(
                b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            );
            let mut antwort = String::new();
            let _ = s.read_to_string(&mut antwort);
            if antwort.contains("LocalTranscript") && antwort.contains("\"ok\"") {
                return true;
            }
        }
        if Instant::now() >= frist {
            return false;
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

fn port_halter() -> Option<u32> {
    let out = std::process::Command::new("/usr/sbin/lsof")
        .args(["-ti", &format!("tcp:{}", port()), "-sTCP:LISTEN"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn ist_eigenes_backend(pid: u32) -> bool {
    let Ok(out) = std::process::Command::new("/bin/ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
    else {
        return false;
    };
    let cmd = String::from_utf8_lossy(&out.stdout);
    cmd.contains("uvicorn") && cmd.contains("localtranscript")
}

fn beende_pid(pid: u32) {
    let _ = std::process::Command::new("/bin/kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(200));
        let lebt = std::process::Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !lebt {
            return;
        }
    }
    let _ = std::process::Command::new("/bin/kill")
        .args(["-KILL", &pid.to_string()])
        .status();
}

/// Bundle-venv relozierbar machen: pyvenv.cfg zeigt auf den
/// python-runtime NEBEN dem venv — absolut, zur Laufzeit gesetzt
/// (die App kann irgendwo installiert sein; v1-Electron-Muster).
fn venv_fixen(resources: &Path) -> Result<(), String> {
    let cfg = resources.join("venv/pyvenv.cfg");
    let runtime = resources.join("python-runtime/bin");
    let alt = std::fs::read_to_string(&cfg).unwrap_or_default();
    let version = alt
        .lines()
        .find(|z| z.starts_with("version"))
        .unwrap_or("version = 3.13.13")
        .to_string();
    let neu = format!(
        "home = {}\ninclude-system-site-packages = false\n{}\nexecutable = {}\n",
        runtime.display(),
        version,
        runtime.join("python3").display()
    );
    if alt.trim() == neu.trim() {
        return Ok(());
    }
    std::fs::write(&cfg, &neu).map_err(|e| {
        format!(
            "pyvenv.cfg nicht schreibbar ({e}) — liegt die App auf einem \
             read-only-Volume oder in der Gatekeeper-Translokation? \
             Einmal nach /Applications kopieren und dort öffnen."
        )
    })
}

struct Backend {
    python: PathBuf,
    cwd: PathBuf,
    bundled: Option<PathBuf>, // Resources-Wurzel im Bundle
}

fn backend_finden(app: &tauri::AppHandle) -> Result<Backend, String> {
    // Bundle: Resources/venv (+ python-runtime, bin, lib, models)
    if let Ok(res) = app.path().resource_dir() {
        let py = res.join("venv/bin/python3");
        if py.is_file() {
            venv_fixen(&res)?;
            return Ok(Backend {
                python: py,
                cwd: res.clone(),
                bundled: Some(res),
            });
        }
    }
    // Dev: Repo-Checkout (Pfad zur Bauzeit eingebrannt, enrich-Muster)
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .ok_or("Repo-Wurzel nicht bestimmbar")?;
    let py = repo.join("backend/.venv/bin/python");
    if py.is_file() {
        return Ok(Backend {
            python: py,
            cwd: repo.join("backend"),
            bundled: None,
        });
    }
    Err(format!(
        "Kein Python-Backend gefunden (weder Bundle-venv noch {})",
        py.display()
    ))
}

#[tauri::command]
async fn backend_starten(
    app: tauri::AppHandle,
    eigen: tauri::State<'_, EigenesBackend>,
) -> Result<(), String> {
    // Läuft schon etwas Gesundes? Benutzen — und prüfen, ob es das
    // eigene Waisenkind aus einem abgestürzten Lauf ist: dann geht es
    // beim Quit mit, sonst bleibt es unangetastet (Terminal-Dev).
    if health_antwortet(0) {
        if let Some((app_pid, b_pid, p)) = merk_lesen() {
            if pid_lebt(app_pid) && app_pid != std::process::id() {
                // Eine ANDERE Instanz dieser App lebt und besitzt das
                // Backend: fremd. Nicht übernehmen, Merkdatei ist ihre.
            } else if p == port()
                && Some(b_pid) == port_halter()
                && ist_eigenes_backend(b_pid)
            {
                // Waise eines abgestürzten Laufs — übernehmen und den
                // Besitz auf UNS umschreiben.
                if let Ok(mut g) = eigen.0.lock() {
                    *g = Some(b_pid);
                }
                merk_schreiben(b_pid);
            } else {
                // Eintrag zeigt ins Leere (PID neu vergeben, Port
                // gewechselt) — weg damit, bevor er jemanden verwirrt.
                merk_loeschen();
            }
        }
        return Ok(());
    }
    // Zombie auf dem Port? Nur ps-identifizierte eigene beenden.
    if let Some(pid) = port_halter() {
        if ist_eigenes_backend(pid) {
            beende_pid(pid);
        } else {
            return Err(format!(
                "Port {} ist von einem fremden Prozess belegt (PID {pid})", port()
            ));
        }
    }
    let b = backend_finden(&app)?;
    let mut cmd = std::process::Command::new(&b.python);
    cmd.args([
        "-m",
        "uvicorn",
        "localtranscript.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        &port().to_string(),
    ])
    .current_dir(&b.cwd)
    .env("PYTHONUNBUFFERED", "1")
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null());
    if let Some(res) = &b.bundled {
        cmd.env("LT_BUNDLED", "1").env("LT_APP_ROOT", res);
    }
    let kind = cmd.spawn().map_err(|e| format!("Spawn: {e}"))?;
    let pid = kind.id();
    if let Ok(mut g) = eigen.0.lock() {
        *g = Some(pid);
    }
    merk_schreiben(pid);
    // Erstes Laden im Bundle importiert torch — großzügige Frist.
    if health_antwortet(90) {
        Ok(())
    } else {
        Err("Backend antwortet nicht (90 s) — Log: Konsole.app".into())
    }
}

#[tauri::command]
fn ordner_oeffnen(pfad: String) -> Result<(), String> {
    std::process::Command::new("/usr/bin/open")
        .arg(&pfad)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Nur das Nötige (User 2026-09-09): das Standardmenü schleppte File,
/// View, Help und ein Services-Untermenü mit — leer oder mit fremden
/// Entwickler-Werkzeugen gefüllt. Bleiben: „Über/Ausblenden/Beenden",
/// die Textbefehle (die App ist ein Editor — ⌘Z/⌘X/⌘C/⌘V sind Pflicht)
/// und die Fensterbefehle. Beschriftungen kommen von macOS und folgen
/// Englisch wie die vordefinierten Einträge (About/Hide/Quit,
/// Undo/Cut/Copy liefert muda auf Englisch) — deutsche
/// Untermenü-Titel daneben wären ein Sprachmischmasch. Das
/// Menü folgt damit NICHT der Oberflächensprache der App.
fn menue(handle: &tauri::AppHandle) -> tauri::Result<tauri::menu::Menu<tauri::Wry>> {
    use tauri::menu::{AboutMetadata, Menu, PredefinedMenuItem, Submenu};
    let ueber = AboutMetadata {
        name: Some("LocalTranscript".into()),
        version: Some(env!("CARGO_PKG_VERSION").into()),
        license: Some("GPL-3.0-or-later".into()),
        website: Some("https://bias.city/b-ias-basel-institut-fuer-angewandte-stadtforschung/".into()),
        website_label: Some("BIAS.City".into()),
        comments: Some(
            "Vollständig lokale Transkription mit Sprecherkennung.".into(),
        ),
        ..Default::default()
    };
    let app = Submenu::with_items(
        handle,
        "LocalTranscript",
        true,
        &[
            &PredefinedMenuItem::about(handle, None, Some(ueber))?,
            &PredefinedMenuItem::separator(handle)?,
            &PredefinedMenuItem::hide(handle, None)?,
            &PredefinedMenuItem::separator(handle)?,
            &PredefinedMenuItem::quit(handle, None)?,
        ],
    )?;
    let text = Submenu::with_items(
        handle,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(handle, None)?,
            &PredefinedMenuItem::redo(handle, None)?,
            &PredefinedMenuItem::separator(handle)?,
            &PredefinedMenuItem::cut(handle, None)?,
            &PredefinedMenuItem::copy(handle, None)?,
            &PredefinedMenuItem::paste(handle, None)?,
            &PredefinedMenuItem::select_all(handle, None)?,
        ],
    )?;
    let fenster = Submenu::with_items(
        handle,
        "Window",
        true,
        &[
            &PredefinedMenuItem::minimize(handle, None)?,
            &PredefinedMenuItem::fullscreen(handle, None)?,
            &PredefinedMenuItem::separator(handle)?,
            &PredefinedMenuItem::close_window(handle, None)?,
        ],
    )?;
    Menu::with_items(handle, &[&app, &text, &fenster])
}

pub fn run() {
    let app = tauri::Builder::default()
        .menu(menue)
        .plugin(tauri_plugin_dialog::init())
        .manage(EigenesBackend(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![backend_starten, ordner_oeffnen])
        .build(tauri::generate_context!())
        .expect("LocalTranscript konnte nicht starten");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            // NUR das selbst gestartete Backend beenden (eiserne Regel);
            // zusätzlich ps-identifizierte eigene Waisen auf dem Port.
            let eigen = handle
                .state::<EigenesBackend>()
                .0
                .lock()
                .ok()
                .and_then(|g| *g);
            // NUR Selbstgestartetes (eiserne Regel): ein im Terminal
            // gestartetes Dev-Backend gehört dem User, nie der App.
            if let Some(pid) = eigen {
                beende_pid(pid);
            }
            // Nur die eigene Merkdatei räumen — gehört sie einer
            // anderen laufenden Instanz, bleibt sie stehen.
            if merk_ist_meine() {
                merk_loeschen();
            }
        }
    });
}
