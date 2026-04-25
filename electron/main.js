'use strict';

const { app, BrowserWindow, Menu, shell, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const net = require('node:net');
const path = require('node:path');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const http = require('node:http');

app.setName('LocalTranscript');

// --- Config -----------------------------------------------------------------

// Layout differs between dev and packaged builds:
//   dev:       PROJECT_ROOT/{backend,venv,bin,models}
//   packaged:  process.resourcesPath/{backend,venv,bin,models,python-runtime,.env,BUNDLED}
const DEV_PROJECT_ROOT = path.resolve(__dirname, '..');
const RES_DIR_DEV = path.join(DEV_PROJECT_ROOT, 'electron', 'resources');

function resolvePaths() {
  if (app.isPackaged) {
    const res = process.resourcesPath;
    return {
      cwd: res,
      python: path.join(res, 'venv', 'bin', 'python'),
      env: { WHISPER_BUNDLED: '1' },
    };
  }
  // Dev mode: prefer the project venv. If a built bundle exists in
  // electron/resources/, allow opting into it via WHISPER_USE_BUNDLE=1
  // for testing the packaged layout end-to-end.
  if (process.env.WHISPER_USE_BUNDLE === '1' && fs.existsSync(RES_DIR_DEV)) {
    return {
      cwd: RES_DIR_DEV,
      python: path.join(RES_DIR_DEV, 'venv', 'bin', 'python'),
      env: { WHISPER_BUNDLED: '1' },
    };
  }
  return {
    cwd: DEV_PROJECT_ROOT,
    python: process.env.WHISPER_PYTHON || path.join(DEV_PROJECT_ROOT, 'venv', 'bin', 'python'),
    env: {},
  };
}

const RUNTIME = resolvePaths();
const PYTHON_BIN = RUNTIME.python;
const BACKEND_CWD = RUNTIME.cwd;
const BACKEND_ENV_OVERRIDES = RUNTIME.env;

const HEALTH_TIMEOUT_MS = 30_000;
const HEALTH_INTERVAL_MS = 250;

// --- State ------------------------------------------------------------------

/** @type {import('electron').BrowserWindow | null} */
let mainWindow = null;
/** @type {import('node:child_process').ChildProcess | null} */
let backendProc = null;
let backendPort = null;
let backendStopping = false;

// --- Helpers ----------------------------------------------------------------

// --- Config persistence -----------------------------------------------------

function configPath() {
  return path.join(app.getPath('userData'), 'config.json');
}

function loadConfig() {
  try {
    return JSON.parse(fs.readFileSync(configPath(), 'utf-8'));
  } catch {
    return {};
  }
}

function saveConfig(cfg) {
  const file = configPath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(cfg, null, 2), 'utf-8');
}

function defaultTranscriptsRoot() {
  return path.join(app.getPath('documents'), 'LocalTranscript');
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

// Sanitize a filename to avoid path traversal — never trust the renderer.
function safeName(name) {
  const base = path.basename(String(name || ''));
  return base.replace(/[\x00-\x1f\\/]/g, '_').slice(0, 200) || 'untitled';
}

// Returns true if `child` is inside `parent` (after resolving).
function isInside(parent, child) {
  const rel = path.relative(parent, child);
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel);
}

// --- IPC handlers -----------------------------------------------------------

function registerIpc() {
  ipcMain.handle('config:get', () => {
    const cfg = loadConfig();
    // Drop a stale path (folder may have been moved/deleted)
    if (cfg.transcriptsRoot && !fs.existsSync(cfg.transcriptsRoot)) {
      delete cfg.transcriptsRoot;
    }
    return cfg;
  });

  ipcMain.handle('config:set-transcripts-root', async (_event, mode) => {
    let chosen = null;
    if (mode === 'default') {
      chosen = defaultTranscriptsRoot();
      ensureDir(chosen);
    } else {
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Speicherort für Transkripte wählen',
        defaultPath: app.getPath('documents'),
        properties: ['openDirectory', 'createDirectory'],
        buttonLabel: 'Auswählen',
      });
      if (result.canceled || !result.filePaths[0]) {
        return { canceled: true };
      }
      chosen = result.filePaths[0];
    }
    const cfg = { ...loadConfig(), transcriptsRoot: chosen };
    saveConfig(cfg);
    if (mainWindow) {
      mainWindow.webContents.send('config:changed', cfg);
    }
    return { transcriptsRoot: chosen };
  });

  ipcMain.handle('run:create-folder', async (_event, folderName) => {
    const cfg = loadConfig();
    if (!cfg.transcriptsRoot) {
      throw new Error('Kein Speicherort konfiguriert');
    }
    const root = cfg.transcriptsRoot;
    if (!fs.existsSync(root)) ensureDir(root);

    const base = safeName(folderName);
    let candidate = path.join(root, base);
    let counter = 2;
    while (fs.existsSync(candidate)) {
      candidate = path.join(root, `${base}_${counter}`);
      counter++;
    }
    ensureDir(candidate);
    return { path: candidate };
  });

  ipcMain.handle('run:copy', async (_event, payload) => {
    const { runFolder, files } = payload || {};
    if (!runFolder || !Array.isArray(files)) {
      throw new Error('Ungültige Kopier-Anfrage');
    }
    const cfg = loadConfig();
    if (!cfg.transcriptsRoot || !isInside(cfg.transcriptsRoot, runFolder)) {
      throw new Error('Run-Ordner liegt nicht unter dem konfigurierten Speicherort');
    }
    if (!fs.existsSync(runFolder)) {
      throw new Error('Run-Ordner existiert nicht mehr');
    }

    const usedNames = new Set(await fsp.readdir(runFolder).catch(() => []));
    const copied = [];
    for (const item of files) {
      if (!item || !item.src || !item.name) continue;
      if (!fs.existsSync(item.src)) continue;

      let destName = safeName(item.name);
      // Avoid clobbering existing entries from prior files in the same batch
      if (usedNames.has(destName)) {
        const ext = path.extname(destName);
        const stem = destName.slice(0, destName.length - ext.length);
        let i = 2;
        while (usedNames.has(`${stem}_${i}${ext}`)) i++;
        destName = `${stem}_${i}${ext}`;
      }
      usedNames.add(destName);

      const dest = path.join(runFolder, destName);
      await fsp.copyFile(item.src, dest);
      copied.push(destName);
    }
    return { copied };
  });

  ipcMain.handle('shell:open-folder', async (_event, folderPath) => {
    if (!folderPath || !fs.existsSync(folderPath)) {
      throw new Error('Ordner nicht gefunden');
    }
    const err = await shell.openPath(folderPath);
    if (err) throw new Error(err);
    return { ok: true };
  });

  ipcMain.handle('shell:open-external', async (_event, url) => {
    if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) {
      throw new Error('Nur http(s)-URLs erlaubt');
    }
    await shell.openExternal(url);
    return { ok: true };
  });

  ipcMain.handle('shell:open-licenses', async () => {
    // Bundled at Resources/THIRD-PARTY-LICENSES.md (packaged) or project root (dev).
    const candidates = app.isPackaged
      ? [path.join(process.resourcesPath, 'THIRD-PARTY-LICENSES.md')]
      : [
          path.join(DEV_PROJECT_ROOT, 'THIRD-PARTY-LICENSES.md'),
          path.join(RES_DIR_DEV, 'THIRD-PARTY-LICENSES.md'),
        ];
    for (const file of candidates) {
      if (fs.existsSync(file)) {
        const err = await shell.openPath(file);
        if (err) throw new Error(err);
        return { ok: true, path: file };
      }
    }
    throw new Error('THIRD-PARTY-LICENSES.md nicht gefunden');
  });
}

// --- Helpers ----------------------------------------------------------------

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function waitForHealth(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(
        { host: '127.0.0.1', port, path: '/api/health', timeout: 1000 },
        (res) => {
          if (res.statusCode === 200) {
            res.resume();
            resolve();
          } else {
            res.resume();
            retry();
          }
        }
      );
      req.on('error', retry);
      req.on('timeout', () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() > deadline) {
        reject(new Error(`Backend did not become ready within ${timeoutMs}ms`));
      } else {
        setTimeout(tick, HEALTH_INTERVAL_MS);
      }
    };
    tick();
  });
}

function pythonExists() {
  try {
    return fs.statSync(PYTHON_BIN).isFile();
  } catch {
    return false;
  }
}

function spawnBackend(port) {
  const args = [
    '-m', 'uvicorn',
    'backend.main:app',
    '--host', '127.0.0.1',
    '--port', String(port),
  ];
  const child = spawn(PYTHON_BIN, args, {
    cwd: BACKEND_CWD,
    env: { ...process.env, PYTHONUNBUFFERED: '1', ...BACKEND_ENV_OVERRIDES },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  child.stdout.on('data', (buf) => process.stdout.write(`[backend] ${buf}`));
  child.stderr.on('data', (buf) => process.stderr.write(`[backend] ${buf}`));
  child.on('exit', (code, signal) => {
    if (!backendStopping) {
      console.error(`[electron] backend exited unexpectedly (code=${code}, signal=${signal})`);
    }
    backendProc = null;
  });

  return child;
}

function stopBackend() {
  if (!backendProc) return;
  backendStopping = true;
  try {
    backendProc.kill('SIGTERM');
  } catch (e) {
    console.error('[electron] failed to SIGTERM backend:', e);
  }
  // Force-kill if it doesn't exit within 4s
  setTimeout(() => {
    if (backendProc) {
      try { backendProc.kill('SIGKILL'); } catch {}
    }
  }, 4000);
}

// --- Window + Menu ----------------------------------------------------------

function buildMenu() {
  const isMac = process.platform === 'darwin';
  const template = [
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about', label: `Über ${app.name}` },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide', label: `${app.name} ausblenden` },
        { role: 'hideOthers', label: 'Andere ausblenden' },
        { role: 'unhide', label: 'Alle einblenden' },
        { type: 'separator' },
        { role: 'quit', label: `${app.name} beenden` },
      ],
    }] : []),
    {
      label: 'Datei',
      submenu: [
        {
          label: 'Neue Transkription',
          accelerator: 'CmdOrCtrl+N',
          click: () => mainWindow && mainWindow.webContents.reload(),
        },
        {
          label: 'Speicherort ändern…',
          click: async () => {
            if (!mainWindow) return;
            const result = await dialog.showOpenDialog(mainWindow, {
              title: 'Neuer Speicherort für Transkripte',
              defaultPath: app.getPath('documents'),
              properties: ['openDirectory', 'createDirectory'],
              buttonLabel: 'Auswählen',
            });
            if (result.canceled || !result.filePaths[0]) return;
            const cfg = { ...loadConfig(), transcriptsRoot: result.filePaths[0] };
            saveConfig(cfg);
            mainWindow.webContents.send('config:changed', cfg);
          },
        },
        {
          label: 'Speicherordner öffnen',
          click: () => {
            const cfg = loadConfig();
            if (cfg.transcriptsRoot && fs.existsSync(cfg.transcriptsRoot)) {
              shell.openPath(cfg.transcriptsRoot);
            }
          },
        },
        { type: 'separator' },
        isMac ? { role: 'close', label: 'Fenster schließen' } : { role: 'quit', label: 'Beenden' },
      ],
    },
    {
      label: 'Bearbeiten',
      submenu: [
        { role: 'undo', label: 'Rückgängig' },
        { role: 'redo', label: 'Wiederherstellen' },
        { type: 'separator' },
        { role: 'cut', label: 'Ausschneiden' },
        { role: 'copy', label: 'Kopieren' },
        { role: 'paste', label: 'Einsetzen' },
        { role: 'selectAll', label: 'Alles auswählen' },
      ],
    },
    {
      label: 'Ansicht',
      submenu: [
        { role: 'reload', label: 'Neu laden' },
        { role: 'forceReload', label: 'Hart neu laden' },
        { role: 'toggleDevTools', label: 'Entwicklerwerkzeuge' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'Originalgröße' },
        { role: 'zoomIn', label: 'Vergrößern' },
        { role: 'zoomOut', label: 'Verkleinern' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'Vollbild' },
      ],
    },
    {
      label: 'Fenster',
      submenu: [
        { role: 'minimize', label: 'Minimieren' },
        { role: 'zoom', label: 'Zoomen' },
        ...(isMac ? [
          { type: 'separator' },
          { role: 'front', label: 'Alle nach vorn' },
        ] : [{ role: 'close', label: 'Schließen' }]),
      ],
    },
    {
      role: 'help',
      label: 'Hilfe',
      submenu: [
        {
          label: 'Projekt auf GitHub',
          click: () => shell.openExternal('https://github.com/BenPohlBasel/LocalTranscript'),
        },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 800,
    minWidth: 720,
    minHeight: 560,
    backgroundColor: '#1f2024',
    titleBarStyle: 'hiddenInset',
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.once('ready-to-show', () => mainWindow.show());
  mainWindow.on('closed', () => { mainWindow = null; });

  // Forward renderer console to the main-process terminal — useful in dev,
  // harmless in prod where stdout isn't visible.
  mainWindow.webContents.on('console-message', (_e, level, message, line, sourceId) => {
    if (level >= 2) {
      const prefix = level === 3 ? '[renderer ERROR]' : '[renderer WARN]';
      console.log(`${prefix} ${sourceId}:${line} ${message}`);
    }
  });

  mainWindow.loadURL(`http://127.0.0.1:${port}/`);
}

// --- App lifecycle ----------------------------------------------------------

async function bootstrap() {
  if (!pythonExists()) {
    const where = app.isPackaged
      ? 'Das Bundle ist beschädigt — Python-Runtime fehlt in den Resources.'
      : 'Im Dev-Modus muss das venv im Projektordner existieren.\n' +
        'Setze WHISPER_PYTHON, um einen anderen Python-Pfad zu verwenden,\n' +
        'oder WHISPER_USE_BUNDLE=1, um gegen ein gebautes Bundle zu starten.';
    dialog.showErrorBox(
      'Python-Backend nicht gefunden',
      `Erwartet: ${PYTHON_BIN}\n\n${where}`
    );
    app.quit();
    return;
  }

  try {
    registerIpc();

    backendPort = await findFreePort();
    console.log(`[electron] using backend port ${backendPort}`);

    backendProc = spawnBackend(backendPort);
    await waitForHealth(backendPort, HEALTH_TIMEOUT_MS);

    buildMenu();
    createWindow(backendPort);
  } catch (err) {
    console.error('[electron] bootstrap failed:', err);
    dialog.showErrorBox('Backend-Start fehlgeschlagen', String(err.message || err));
    stopBackend();
    app.quit();
  }
}

app.whenReady().then(bootstrap);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWindow === null && backendPort !== null) {
    createWindow(backendPort);
  }
});

app.on('before-quit', stopBackend);
process.on('exit', stopBackend);
