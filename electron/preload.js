'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronApi', {
  isElectron: true,

  // Config
  getConfig: () => ipcRenderer.invoke('config:get'),
  setTranscriptsRoot: (mode) => ipcRenderer.invoke('config:set-transcripts-root', mode),
  onConfigChanged: (cb) => {
    const listener = (_event, data) => cb(data);
    ipcRenderer.on('config:changed', listener);
    return () => ipcRenderer.removeListener('config:changed', listener);
  },

  // Per-run folder + autosave
  createRunFolder: (folderName) => ipcRenderer.invoke('run:create-folder', folderName),
  copyToRun: (runFolder, files) => ipcRenderer.invoke('run:copy', { runFolder, files }),

  // Misc
  openFolder: (folderPath) => ipcRenderer.invoke('shell:open-folder', folderPath),
  openLicenses: () => ipcRenderer.invoke('shell:open-licenses'),
  openExternal: (url) => ipcRenderer.invoke('shell:open-external', url),
});
