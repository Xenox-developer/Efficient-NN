"""Compile the editable LaTeX report without changing saved measurements."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='External report.tex with its figure directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        parser.error(f'LaTeX source not found: {source}')
    output = (args.output or root.parent / 'report.pdf').resolve()
    compiler = shutil.which('pdflatex')
    if compiler is None:
        mac_compiler = Path('/Library/TeX/texbin/pdflatex')
        compiler = str(mac_compiler) if mac_compiler.is_file() else None
    if compiler is None:
        parser.error('pdflatex is required. Install TeX Live or MacTeX with Babel Russian and CM-Super.')
    with tempfile.TemporaryDirectory(prefix='hw2-report-') as build_dir:
        for _ in range(2):
            result = subprocess.run(
                [compiler, '-interaction=nonstopmode', '-halt-on-error',
                 f'-output-directory={build_dir}', source.name],
                cwd=source.parent, capture_output=True, text=True, errors='replace')
            if result.returncode:
                raise SystemExit(result.stdout + result.stderr)
        pdf = Path(build_dir) / f'{source.stem}.pdf'
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, output)
    print(output)


if __name__ == '__main__':
    main()
