"""Build a minimal, reproducible arXiv source archive with full-line notes removed."""
import argparse
import gzip
from hashlib import sha256
from pathlib import Path
import json
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = (here/"main.tex").read_text()
    clean = "".join(line for line in source.splitlines(keepends=True)
                    if not line.lstrip().startswith("%"))
    (args.output_dir/"main.tex").write_text("\\def\\PREPRINT{}\n"+clean)
    for original, name in [(here/"main-preprint.bbl", "main.bbl"),
                           (here/"corl_2026.sty", "corl_2026.sty"),
                           (here.parent/"fig_noise_floor.pdf", "fig_noise_floor.pdf")]:
        shutil.copyfile(original, args.output_dir/name)
    names = ("main.tex", "main.bbl", "corl_2026.sty", "fig_noise_floor.pdf")
    # Preserve the third-party style, including its license comments.
    identities = {name: {"bytes": (args.output_dir/name).stat().st_size,
                         "sha256": sha256((args.output_dir/name).read_bytes()).hexdigest()}
                  for name in names}
    archive = here/"arxiv.tar.gz"
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb",
                                                fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name in names:
                path = args.output_dir/name
                info = tar.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
    (here/"arxiv-manifest.json").write_text(json.dumps({
        "files": identities, "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
        "full_line_notes_removed": sum(line.lstrip().startswith("%")
                                      for line in source.splitlines()),
        "third_party_license_comments_preserved": True,
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
