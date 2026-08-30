// LocalTranscript-Shell (enrich-Muster, vereinfacht): die Shell ist dumm —
// sie spawnt das Python-Backend auf Port 44100, wartet auf /api/health
// und beendet beim Quit NUR, was sie selbst gestartet hat. Ein fremd
// gestartetes gesundes Backend (Terminal-Dev) wird benutzt, nie angefasst.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};

const PORT: u16 = 44100;

struct EigenesBackend(Mutex<Option<u32>>);

fn health_antwortet(frist_s: u64) -> bool {
    let frist = Instant::now() + Duration::from_secs(frist_s);
    loop {
        if let Ok(mut s) = TcpStream::connect_timeout(
            &format!("127.0.0.1:{PORT}").parse().unwrap(),
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
        .args(["-ti", &format!("tcp:{PORT}"), "-sTCP:LISTEN"])
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
fn venv_fixen(resources: &Path) {
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
    let _ = std::fs::write(&cfg, neu);
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
            venv_fixen(&res);
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
fn backend_starten(
    app: tauri::AppHandle,
    eigen: tauri::State<'_, EigenesBackend>,
) -> Result<(), String> {
    // Läuft schon etwas Gesundes? Benutzen (nie anfassen).
    if health_antwortet(0) {
        return Ok(());
    }
    // Zombie auf dem Port? Nur ps-identifizierte eigene beenden.
    if let Some(pid) = port_halter() {
        if ist_eigenes_backend(pid) {
            beende_pid(pid);
        } else {
            return Err(format!(
                "Port {PORT} ist von einem fremden Prozess belegt (PID {pid})"
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
        &PORT.to_string(),
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

pub fn run() {
    let app = tauri::Builder::default()
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
            if let Some(pid) = eigen {
                beende_pid(pid);
            }
            if let Some(pid) = port_halter() {
                if ist_eigenes_backend(pid) {
                    beende_pid(pid);
                }
            }
        }
    });
}
