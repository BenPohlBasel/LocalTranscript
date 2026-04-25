# Third-Party Software Notices and Licenses

`LocalTranscript` is distributed as a self-contained macOS application bundle. The bundle ships several third-party components. This file lists every component, its upstream project, the license it is distributed under, and where to obtain the canonical license text.

The `LocalTranscript` application code itself is licensed under the **MIT License** (see `LICENSE`).

## How to read this file

Each entry below has the form:

> **Name** — *License (SPDX)*
> Project URL · License URL · Used for: …

Where the license text is short (MIT/BSD/ISC), it is included verbatim once at the bottom of this file with the relevant copyright holder. Where the license is longer (Apache 2.0, LGPL, MPL, PSF), the canonical full text on the project's site is the authoritative copy.

---

## 1. Native binaries (bundled in `Resources/bin` and `Resources/lib`)

### whisper.cpp / whisper-cli + libwhisper, libggml*
- *MIT License (SPDX: MIT)*
- Project: <https://github.com/ggerganov/whisper.cpp>
- Copyright © 2022–2026 Georgi Gerganov, ggml.ai
- License: <https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE>
- Used for: speech-to-text inference via the bundled `whisper-cli` and its `libwhisper.dylib`, `libggml.dylib`, `libggml-cpu.dylib`, `libggml-blas.dylib`, `libggml-metal.dylib`, `libggml-base.dylib` (located in `Resources/lib`).

### ffmpeg (via ffmpeg-static npm package)
- *GNU Lesser General Public License v2.1 or later (SPDX: LGPL-2.1-or-later)*
- Project: <https://ffmpeg.org/>
- License: <https://ffmpeg.org/legal.html> · <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
- Source: <https://git.ffmpeg.org/ffmpeg.git>
- Used for: audio format conversion. The bundled `ffmpeg` binary in `Resources/bin/ffmpeg` is the LGPL build provided by the [`ffmpeg-static`](https://github.com/eugeneware/ffmpeg-static) npm package. We invoke it as a subprocess; we do not statically link it into our code.

  **LGPL compliance note:** the bundled binary is dynamically linkable. If you wish to replace it with your own ffmpeg build of the same major version, drop the new binary into `Resources/bin/ffmpeg` of the unpacked `.app` and the application will pick it up. Source code for the exact ffmpeg version bundled is available from <https://ffmpeg.org/download.html>.

---

## 2. Models (bundled in `Resources/models`)

### Whisper large-v3-turbo (`ggml-large-v3-turbo.bin`)
- *MIT License (SPDX: MIT)*
- Project: <https://github.com/openai/whisper>
- Model card: <https://huggingface.co/openai/whisper-large-v3-turbo>
- ggml conversion: <https://huggingface.co/ggerganov/whisper.cpp/tree/main>
- Copyright © 2022 OpenAI
- License: <https://github.com/openai/whisper/blob/main/LICENSE>
- Used for: automatic speech recognition.

### SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)
- *Apache License 2.0 (SPDX: Apache-2.0)*
- Model card: <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb>
- Project: <https://speechbrain.github.io/>
- License: <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/main/LICENSE>
- Used for: speaker embedding extraction during diarization.
- Trained on the [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset, released by the SpeechBrain authors under Apache 2.0.

### Silero VAD model
- *MIT License (SPDX: MIT)*
- Project: <https://github.com/snakers4/silero-vad>
- License: <https://github.com/snakers4/silero-vad/blob/master/LICENSE>
- Used for: voice activity detection. The model file is bundled inside the `silero-vad` Python package.

---

## 3. Python runtime

### CPython 3.13
- *Python Software Foundation License 2.0 (SPDX: PSF-2.0)*
- Project: <https://www.python.org/>
- License: <https://docs.python.org/3.13/license.html>
- Distribution: relocatable build by [astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone) (MPL 2.0 build infrastructure, PSF-licensed binaries). Project: <https://github.com/astral-sh/python-build-standalone>

---

## 4. Python dependencies (in `Resources/venv`)

### Speech / ML

#### speechbrain
- *Apache License 2.0 (SPDX: Apache-2.0)*
- Project: <https://github.com/speechbrain/speechbrain>
- License: <https://github.com/speechbrain/speechbrain/blob/develop/LICENSE>
- Used for: speaker embedding model loading and inference.

#### hyperpyyaml
- *Apache License 2.0*
- Project: <https://github.com/speechbrain/HyperPyYAML>
- License: <https://github.com/speechbrain/HyperPyYAML/blob/main/LICENSE>

#### silero-vad
- *MIT License*
- Project: <https://github.com/snakers4/silero-vad>
- License: <https://github.com/snakers4/silero-vad/blob/master/LICENSE>

#### torch (PyTorch) and torchaudio
- *BSD-style (SPDX: BSD-3-Clause)*
- Project: <https://pytorch.org/>
- License: <https://github.com/pytorch/pytorch/blob/main/LICENSE>
- torchaudio: <https://github.com/pytorch/audio/blob/main/LICENSE>

#### scikit-learn
- *BSD-3-Clause*
- Project: <https://scikit-learn.org/>
- License: <https://github.com/scikit-learn/scikit-learn/blob/main/COPYING>

#### NumPy
- *BSD-3-Clause*
- Project: <https://numpy.org/>
- License: <https://github.com/numpy/numpy/blob/main/LICENSE.txt>

#### SciPy
- *BSD-3-Clause*
- Project: <https://scipy.org/>
- License: <https://github.com/scipy/scipy/blob/main/LICENSE.txt>

#### sympy, mpmath
- *BSD-3-Clause*
- Project: <https://www.sympy.org/> · <https://mpmath.org/>
- License: <https://github.com/sympy/sympy/blob/master/LICENSE> · <https://github.com/mpmath/mpmath/blob/master/LICENSE>

#### networkx
- *BSD-3-Clause*
- Project: <https://networkx.org/>
- License: <https://github.com/networkx/networkx/blob/main/LICENSE.txt>

#### joblib, threadpoolctl
- *BSD-3-Clause*
- Project: <https://joblib.readthedocs.io/> · <https://github.com/joblib/threadpoolctl>
- License: <https://github.com/joblib/joblib/blob/main/LICENSE.txt>

#### sentencepiece
- *Apache License 2.0*
- Project: <https://github.com/google/sentencepiece>
- License: <https://github.com/google/sentencepiece/blob/master/LICENSE>

### Web / API server

#### FastAPI
- *MIT License*
- Project: <https://fastapi.tiangolo.com/>
- License: <https://github.com/fastapi/fastapi/blob/master/LICENSE>

#### Starlette
- *BSD-3-Clause*
- Project: <https://www.starlette.io/>
- License: <https://github.com/encode/starlette/blob/master/LICENSE.md>

#### uvicorn
- *BSD-3-Clause*
- Project: <https://www.uvicorn.org/>
- License: <https://github.com/encode/uvicorn/blob/master/LICENSE.md>

#### uvloop
- *Apache License 2.0 / MIT (dual-licensed)*
- Project: <https://github.com/MagicStack/uvloop>
- License: <https://github.com/MagicStack/uvloop/blob/master/LICENSE-APACHE> · <https://github.com/MagicStack/uvloop/blob/master/LICENSE-MIT>

#### httptools, watchfiles
- *MIT License*
- Project: <https://github.com/MagicStack/httptools> · <https://github.com/samuelcolvin/watchfiles>
- License: <https://github.com/MagicStack/httptools/blob/master/LICENSE> · <https://github.com/samuelcolvin/watchfiles/blob/main/LICENSE>

#### websockets
- *BSD-3-Clause*
- Project: <https://websockets.readthedocs.io/>
- License: <https://github.com/python-websockets/websockets/blob/main/LICENSE>

#### pydantic, pydantic-core, annotated-types
- *MIT License*
- Project: <https://docs.pydantic.dev/>
- License: <https://github.com/pydantic/pydantic/blob/main/LICENSE>

#### python-multipart
- *Apache License 2.0*
- Project: <https://github.com/Kludex/python-multipart>
- License: <https://github.com/Kludex/python-multipart/blob/master/LICENSE.txt>

#### python-dotenv
- *BSD-3-Clause*
- Project: <https://github.com/theskumar/python-dotenv>
- License: <https://github.com/theskumar/python-dotenv/blob/main/LICENSE>

### HTTP client stack

#### httpx, httpcore
- *BSD-3-Clause*
- Project: <https://www.python-httpx.org/>
- License: <https://github.com/encode/httpx/blob/master/LICENSE.md>

#### h11
- *MIT License*
- Project: <https://github.com/python-hyper/h11>
- License: <https://github.com/python-hyper/h11/blob/master/LICENSE.txt>

#### anyio
- *MIT License*
- Project: <https://github.com/agronholm/anyio>
- License: <https://github.com/agronholm/anyio/blob/master/LICENSE>

#### requests, urllib3
- *Apache License 2.0 / MIT*
- Project: <https://requests.readthedocs.io/> · <https://urllib3.readthedocs.io/>
- License: <https://github.com/psf/requests/blob/main/LICENSE> · <https://github.com/urllib3/urllib3/blob/main/LICENSE.txt>

#### certifi
- *Mozilla Public License 2.0 (SPDX: MPL-2.0)*
- Project: <https://github.com/certifi/python-certifi>
- License: <https://github.com/certifi/python-certifi/blob/master/LICENSE>

#### charset-normalizer
- *MIT License*
- Project: <https://github.com/jawah/charset_normalizer>
- License: <https://github.com/jawah/charset_normalizer/blob/master/LICENSE>

#### idna
- *BSD-3-Clause*
- Project: <https://github.com/kjd/idna>
- License: <https://github.com/kjd/idna/blob/master/LICENSE.md>

### HuggingFace Hub

#### huggingface-hub
- *Apache License 2.0*
- Project: <https://github.com/huggingface/huggingface_hub>
- License: <https://github.com/huggingface/huggingface_hub/blob/main/LICENSE>

#### hf-xet
- *Apache License 2.0*
- Project: <https://github.com/huggingface/xet-core>
- License: <https://github.com/huggingface/xet-core/blob/main/LICENSE>

### Audio I/O

#### soundfile, pydub
- *BSD-3-Clause / MIT*
- Project: <https://github.com/bastibe/python-soundfile> · <https://github.com/jiaaro/pydub>
- License: <https://github.com/bastibe/python-soundfile/blob/master/LICENSE> · <https://github.com/jiaaro/pydub/blob/master/LICENSE>

#### cffi, pycparser
- *MIT License / BSD-3-Clause*
- Project: <https://cffi.readthedocs.io/> · <https://github.com/eliben/pycparser>
- License: <https://github.com/python-cffi/cffi/blob/main/LICENSE> · <https://github.com/eliben/pycparser/blob/main/LICENSE>

### Utilities

#### click, jinja2, markupsafe
- *BSD-3-Clause*
- Project: <https://palletsprojects.com/>
- License: <https://github.com/pallets/click/blob/main/LICENSE.txt> · <https://github.com/pallets/jinja/blob/main/LICENSE.txt> · <https://github.com/pallets/markupsafe/blob/main/LICENSE.txt>

#### rich, pygments, markdown-it-py, mdurl
- *MIT / BSD-2-Clause*
- License: <https://github.com/Textualize/rich/blob/master/LICENSE> · <https://github.com/pygments/pygments/blob/master/LICENSE> · <https://github.com/executablebooks/markdown-it-py/blob/master/LICENSE> · <https://github.com/executablebooks/mdurl/blob/main/LICENSE>

#### typer, shellingham
- *MIT / ISC License*
- Project: <https://typer.tiangolo.com/>
- License: <https://github.com/fastapi/typer/blob/master/LICENSE> · <https://github.com/sarugaku/shellingham/blob/master/LICENSE>

#### pyyaml, ruamel.yaml, ruamel.yaml.clib
- *MIT License*
- License: <https://github.com/yaml/pyyaml/blob/main/LICENSE> · <https://sourceforge.net/projects/ruamel-yaml/>

#### packaging
- *Apache License 2.0 / BSD-2-Clause (dual)*
- Project: <https://github.com/pypa/packaging>
- License: <https://github.com/pypa/packaging/blob/main/LICENSE>

#### filelock
- *The Unlicense (Public Domain)*
- Project: <https://github.com/tox-dev/filelock>
- License: <https://github.com/tox-dev/filelock/blob/main/LICENSE>

#### fsspec
- *BSD-3-Clause*
- Project: <https://github.com/fsspec/filesystem_spec>
- License: <https://github.com/fsspec/filesystem_spec/blob/master/LICENSE>

#### tqdm
- *MIT License + MPL 2.0 (dual)*
- Project: <https://github.com/tqdm/tqdm>
- License: <https://github.com/tqdm/tqdm/blob/master/LICENCE>

#### typing-extensions, typing-inspection, annotated-doc
- *PSF / MIT*
- License: <https://github.com/python/typing_extensions/blob/main/LICENSE> · <https://github.com/pydantic/typing-inspection/blob/main/LICENSE>

### macOS bridge (PyWebView legacy support)

#### pywebview, proxy_tools, bottle
- *BSD-3-Clause / MIT*
- Project: <https://pywebview.flowrl.com/>
- License: <https://github.com/r0x0r/pywebview/blob/master/LICENSE>

#### pyobjc-core, pyobjc-framework-Cocoa, -Quartz, -WebKit, -security, -UniformTypeIdentifiers
- *MIT License*
- Project: <https://github.com/ronaldoussoren/pyobjc>
- License: <https://github.com/ronaldoussoren/pyobjc/blob/master/pyobjc-core/License.txt>

---

## 5. Electron / JavaScript runtime

### Electron
- *MIT License* (Electron core); embedded Chromium components have their own licenses listed at runtime in *Electron → About → Credits*.
- Project: <https://www.electronjs.org/>
- License: <https://github.com/electron/electron/blob/main/LICENSE>

### Node.js (bundled inside Electron)
- *MIT License + many third-party licenses for sub-components*
- Project: <https://nodejs.org/>
- License: <https://github.com/nodejs/node/blob/main/LICENSE>

### npm packages used in `electron/main.js` and `electron/preload.js`

The Electron app code only uses Electron core APIs and Node.js built-in modules (`fs`, `path`, `child_process`, `net`, `http`). No additional runtime npm packages are bundled in `app.asar`.

#### Build-time only (not in the shipped bundle)
- `electron-builder` — MIT — <https://github.com/electron-userland/electron-builder/blob/master/LICENSE>
- `ffmpeg-static` — GPL-3.0 (the npm packaging itself; the binary it provides is LGPL ffmpeg) — <https://github.com/eugeneware/ffmpeg-static/blob/master/LICENSE>

---

## 6. Frontend assets

The HTML/CSS/JS in `Resources/frontend` is original work and shipped under the MIT License along with the application.

---

## Verbatim short licenses

The following short license texts apply to multiple components above and are reproduced once here.

### MIT License (generic template)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BSD 3-Clause License (generic template)

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### Apache License 2.0

The full text is at <https://www.apache.org/licenses/LICENSE-2.0> and reproduced in `NOTICE` and the per-project LICENSE files linked above.

### LGPL 2.1

The full text is at <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.

### Mozilla Public License 2.0

The full text is at <https://www.mozilla.org/en-US/MPL/2.0/>.

### Python Software Foundation License 2.0

The full text is at <https://docs.python.org/3.13/license.html>.

---

If you spot a missing attribution or an incorrect license, please file an issue at
<https://github.com/BenPohlBasel/LocalTranscript/issues>.
