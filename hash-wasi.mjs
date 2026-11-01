import { WASI } from 'node:wasi';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { loadavg } from 'node:os';
const [file, input, output] = process.argv.slice(2);
const wasi = new WASI({ version: 'preview1', args: [file, input], env: process.env, preopens: { '/': '/' } });
let memory, hash = createHash('sha256'), bytes = 0;
const pictures = [];
const { instance } = await WebAssembly.instantiate(readFileSync(file), {
  wasi_snapshot_preview1: wasi.wasiImport,
  picture_hash: {
    plane: (ptr, len) => { hash.update(new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0)); bytes += len; },
    end_picture: () => { pictures.push({bytes, sha256: hash.digest('hex')}); hash = createHash('sha256'); bytes = 0; }
  }
});
memory = instance.exports.memory;
wasi.start(instance);
writeFileSync(output, JSON.stringify(pictures, null, 2));
console.log(JSON.stringify({input, frames: pictures.length, load: loadavg(), digest: createHash('sha256').update(JSON.stringify(pictures)).digest('hex')}));
