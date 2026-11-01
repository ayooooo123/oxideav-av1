import { WASI } from 'node:wasi';
import { readFileSync } from 'node:fs';
const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: 'preview1', args: [file, ...args], env: process.env, preopens: { '/': '/' } });
const { instance } = await WebAssembly.instantiate(readFileSync(file), { wasi_snapshot_preview1: wasi.wasiImport });
wasi.start(instance);
