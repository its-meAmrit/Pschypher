# Third-party software

PSCYPHER itself is Apache-2.0 (see `LICENSE`). It uses the packages and services below. Nothing is vendored or
redistributed in this repository. Licences are from general knowledge and should be re-checked against the versions
you install (`pip install pip-licenses && pip-licenses --format=markdown`). This is not legal advice.

| Package / service | Used for | Licence (verify) | Required? |
|---|---|---|---|
| Python, tkinter | runtime, GUI | PSF | yes |
| networkx | graph, centrality, Louvain | BSD-3-Clause | yes |
| numpy | numerics | BSD-3-Clause | yes |
| scikit-learn | link-prediction classifier, metrics | BSD-3-Clause | yes |
| openai (Python SDK) | Token Factory client | Apache-2.0 | yes |
| pydantic | FIR schema validation | MIT | optional |
| Pillow | image handling | MIT-CMU (HPND) | optional |
| pytesseract / Tesseract | local OCR fallback | Apache-2.0 | optional; Tesseract is an external binary |
| neo4j driver | optional graph mirror | Apache-2.0 | optional |
| thefuzz (+ rapidfuzz in recent releases) | fuzzy name matching | MIT | optional |
| **PyMuPDF (`fitz`)** | PDF text and page rendering | **AGPL-3.0 or commercial (Artifex)** | optional in code; needed for the demo FIR PDFs |
| Tavily API | optional web context | service terms | optional |
| Nebius Token Factory | inference | service terms | yes |
| NVIDIA Nemotron models | called via API; weights not distributed | model licence on each model card | yes |

## PyMuPDF (AGPL-3.0)

PyMuPDF is **not** included in this repository or in `requirements*.txt`. It is imported only if you install it yourself,
and it is used to read FIR PDFs and to render pages for the vision model. Installing it is your choice and subject to
its licence. Whether importing an AGPL library at runtime affects the licence of the importing code is a contested
question; we disclose the dependency rather than claim to have resolved it. A permissive replacement (`pypdf` for text,
`pypdfium2` for rendering) is a possible future change and has not been made or tested.

## python-Levenshtein (GPL)

Not used and not required. `thefuzz` works without it.
