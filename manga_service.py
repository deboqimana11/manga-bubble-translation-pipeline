from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from translate_manga import (
    DEFAULT_FONT,
    DEFAULT_SEGMENTATION_FILE,
    DEFAULT_SEGMENTATION_REPO,
    DEFAULT_TRANSLATION_MODEL,
    build_cache_key,
    collect_image_paths,
    create_runtime,
    ensure_dirs,
    process_images,
)


def build_args(
    input_path: str,
    output_path: str,
    force: bool,
    conf: float,
    margin: int,
    font: str,
    inpaint_backend: str,
    text_detector: str,
) -> argparse.Namespace:
    return SimpleNamespace(
        input=input_path,
        output=output_path,
        force=force,
        conf=conf,
        margin=margin,
        font=font,
        segmentation_repo=DEFAULT_SEGMENTATION_REPO,
        segmentation_file=DEFAULT_SEGMENTATION_FILE,
        translation_model=DEFAULT_TRANSLATION_MODEL,
        inpaint_backend=inpaint_backend,
        text_detector=text_detector,
    )


class MangaServiceHandler(BaseHTTPRequestHandler):
    runtime = None
    base_args = None

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_loaded": True,
                    "input": self.base_args.input,
                    "output": self.base_args.output,
                },
            )
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/process":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        args = build_args(
            input_path=str(payload.get("input", self.base_args.input)),
            output_path=str(payload.get("output", self.base_args.output)),
            force=bool(payload.get("force", False)),
            conf=float(payload.get("conf", self.base_args.conf)),
            margin=int(payload.get("margin", self.base_args.margin)),
            font=str(payload.get("font", self.base_args.font)),
            inpaint_backend=str(payload.get("inpaint_backend", self.base_args.inpaint_backend)),
            text_detector=str(payload.get("text_detector", self.base_args.text_detector)),
        )

        input_path = Path(args.input).resolve()
        output_dir = Path(args.output).resolve()
        translated_dir, debug_dir, json_dir = ensure_dirs(output_dir)
        image_paths = collect_image_paths(input_path)
        if not image_paths:
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "no_images", "input": str(input_path)},
            )
            return

        cache_key = build_cache_key(args)
        stats = process_images(
            image_paths=image_paths,
            runtime=self.runtime,
            translated_dir=translated_dir,
            debug_dir=debug_dir,
            json_dir=json_dir,
            args=args,
            cache_key=cache_key,
        )
        self._json_response(
            HTTPStatus.OK,
            {
                "status": "ok",
                "processed": stats["processed"],
                "cached": stats["cached"],
                "input": str(input_path),
                "output": str(output_dir),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent local manga translation service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--input", default=".")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--inpaint-backend", choices=["lama", "opencv"], default="lama")
    parser.add_argument("--text-detector", choices=["comic-text-detector", "none"], default="comic-text-detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_args = build_args(
        input_path=args.input,
        output_path=args.output,
        force=False,
        conf=args.conf,
        margin=args.margin,
        font=args.font,
        inpaint_backend=args.inpaint_backend,
        text_detector=args.text_detector,
    )
    runtime = create_runtime(base_args)
    MangaServiceHandler.runtime = runtime
    MangaServiceHandler.base_args = base_args

    server = ThreadingHTTPServer((args.host, args.port), MangaServiceHandler)
    print(f"Listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
