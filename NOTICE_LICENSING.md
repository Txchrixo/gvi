# Licensing notice

The core GVI package (default `pip install -e .`, no extras) only pulls in
Apache/BSD/MIT-style dependencies (Pillow, numpy, pydantic, typer, rich,
scikit-image, scikit-learn, opencv-python-headless). **Nothing in the
default install is AGPL.**

## The `yolo`, `sam2`, and `full` extras pull in AGPL-3.0 code

`ultralytics` (used both for the YOLO segmenter and as the backend the SAM 2
segmenter's `sam2` extra depends on) is licensed under **AGPL-3.0** by
default, not Apache/MIT like the rest of this project's dependencies.
Ultralytics also offers a commercial license as an alternative — see their
own licensing page before making a decision; this project does not have an
opinion on which option is right for you, only on making sure you know
there's a decision to make.

**Practical implication:** AGPL-3.0 has a network-use clause. If you run a
modified version of software that includes AGPL-licensed code as a network
service that other people can use (e.g. you stand up the GVI REST API or
MCP server publicly, or as a paid/public tool, with the `yolo`/`sam2`/`full`
extras installed), AGPL-3.0 can require you to make the complete
corresponding source of your modified version available to those users,
under AGPL-3.0.

**This does NOT apply if:**
- You install only the core dependencies (no `yolo`/`sam2`/`full` extras) —
  there is no AGPL code in your install at all, full stop.
- You keep your fork private and don't expose it as a network service to
  anyone outside yourself.

**This MIGHT apply if:**
- You install the `yolo`, `sam2`, or `full` extras **and** publish your fork
  publicly **and/or** run it as a service other people can reach over a
  network (including "for a client", "as a SaaS", or "as a Discord/Slack
  bot other people use").

This is general information, not legal advice — if your use case is
ambiguous, get an actual legal opinion before publishing or productizing a
build that includes the AGPL extras.

## Everything else

| Component | License |
|---|---|
| Core GVI code (this repo) | MIT |
| OpenCV / opencv-python-headless | Apache-2.0 |
| Pillow | MIT-CMI (HPND-style) |
| NumPy | BSD-3-Clause |
| scikit-image | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| Pydantic | MIT |
| Typer | MIT |
| Rich | MIT |
| FastAPI (api extra) | MIT |
| PyMuPDF (pdf extra) | AGPL-3.0 **or** commercial license from Artifex |
| cairosvg (svg extra) | LGPL-3.0 |
| psd-tools (psd extra) | MIT |
| ultralytics (yolo/sam2/full extras) | **AGPL-3.0** or commercial |
| easyocr (ocr/full extras) | Apache-2.0 |
| torch / torchvision (sam2/yolo/ocr/full extras) | BSD-3-Clause |

Note that `PyMuPDF` (the `pdf` extra) is **also** AGPL-3.0/commercial-dual-
licensed, same situation as `ultralytics` above — this was not flagged in
the original v1.0.0 README and is worth knowing about if you plan to enable
PDF import in a public/hosted build.
