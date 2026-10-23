"""
Builds the static, server-free version of the web app for GitHub Pages.

The page is the same as the Flask one, but MediaPipe and the
TensorFlow-trained network both run in the browser: the network's weights
are exported from models/mlp.npz to models/mlp.json and evaluated by
web/static/js/classifier.js.

Output (default ../site):
  index.html           rendered from web/templates/index.html
  static/              CSS and JavaScript
  signs/               reference handshape images
  models/mlp.json      network weights for the browser
  models/hand_landmarker.task   MediaPipe's hand model

Usage: python build_site.py [--out ../site]
Preview: python -m http.server -d ../site 8080
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from model_files import MODELS_DIR, ensure_landmarker_model

HERE = Path(__file__).resolve().parent


def export_mlp_json(npz_path, labels):
    """Converts the exported Keras weights into JSON for classifier.js."""
    weights = np.load(npz_path)
    layers = []
    i = 0
    while f"W{i}" in weights:
        layers.append({
            "W": np.round(weights[f"W{i}"].astype(np.float64), 7).tolist(),
            "b": np.round(weights[f"b{i}"].astype(np.float64), 7).tolist(),
        })
        i += 1
    return {
        "labels": labels,
        "feature_version": 2,
        "mean": np.round(weights["mean"].astype(np.float64), 7).tolist(),
        "std": np.round(weights["std"].astype(np.float64), 7).tolist(),
        "layers": layers,
    }


def build(out):
    out = Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)

    meta = json.loads((MODELS_DIR / "labels.json").read_text())
    labels = meta["labels"]

    # Every path is relative, since Pages serves the site from /<repo>/.
    env = Environment(
        loader=FileSystemLoader(HERE / "web" / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("index.html").render(
        labels=labels,
        asset=lambda name: f"static/{name}",
        signs_base="signs/",
        client_config={
            "mode": "static",
            "landmarkerUrl": "models/hand_landmarker.task",
            "mlpUrl": "models/mlp.json",
            "signsBase": "signs/",
        },
    )
    (out / "index.html").write_text(html)

    shutil.copytree(HERE / "web" / "static", out / "static")
    shutil.copytree(HERE / "signImages", out / "signs")
    shutil.copy(ensure_landmarker_model(), out / "models" / "hand_landmarker.task")
    (out / "models" / "mlp.json").write_text(
        json.dumps(export_mlp_json(MODELS_DIR / "mlp.npz", labels), separators=(",", ":"))
    )
    # Serve files as-is; don't let Pages run them through Jekyll.
    (out / ".nojekyll").touch()

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Built static site in {out} ({size_mb:.1f} MB)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=HERE.parent / "site", type=Path)
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
