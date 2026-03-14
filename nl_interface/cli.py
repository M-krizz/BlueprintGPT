"""CLI entrypoint for the natural-language specification interface."""

from __future__ import annotations

import argparse
import json

from nl_interface.runner import DEFAULT_CHECKPOINT, DEFAULT_REGULATION_FILE, execute_response
from nl_interface.service import process_user_request


def main():
    parser = argparse.ArgumentParser(description="Parse natural-language residential requirements into a strict BlueprintGPT spec.")
    parser.add_argument("text", nargs="?", default="", help="Natural-language request to parse.")
    parser.add_argument("--current-spec", help="Optional JSON file containing the current user-facing spec.")
    parser.add_argument("--boundary", help="Boundary rectangle as 'width,height' in metres.")
    parser.add_argument("--area", type=float, help="Optional total area for backend translation.")
    parser.add_argument("--area-unit", default="sq.m", help="Area unit for --area. Defaults to sq.m.")
    parser.add_argument("--entrance-point", help="Optional explicit entrance point as 'x,y'.")
    parser.add_argument("--run", action="store_true", help="Execute the routed backend when the request is backend-ready.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for execution artifacts when --run is used.")
    parser.add_argument("--output-prefix", help="Optional prefix for generated artifact filenames.")
    parser.add_argument("--regulation", default=DEFAULT_REGULATION_FILE, help="Regulation JSON path for execution.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Learned-model checkpoint path for learned execution.")
    parser.add_argument("--device", default="cpu", help="Torch device for learned execution.")
    parser.add_argument("--k", type=int, default=10, help="Learned runner: max raw generation attempts.")
    parser.add_argument("--top-m", type=int, default=3, help="Learned runner: number of shortlisted candidates to repair.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Learned runner temperature.")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="Learned runner top-p sampling value.")
    parser.add_argument("--top-k", type=int, default=0, help="Learned runner top-k sampling value.")
    parser.add_argument("--llm-provider", choices=["gemini"], help="Optional LLM provider for explanations (e.g., gemini).")
    parser.add_argument(
        "--llm-api-key",
        help="API key for the selected LLM provider. If omitted, GEMINI_API_KEY env var is used for Gemini.",
    )
    parser.add_argument("--llm-model", help="Optional LLM model name (provider-specific).")
    args = parser.parse_args()

    current_spec = _load_json(args.current_spec) if args.current_spec else None
    resolution = _build_resolution(args)
    response = process_user_request(args.text, current_spec=current_spec, resolution=resolution)

    exit_code = 0
    if args.run:
        if not response.get("backend_ready"):
            response["execution"] = {
                "status": "blocked",
                "reason": "Request is not backend-ready.",
                "missing_fields": response.get("missing_fields", []),
                "validation_errors": response.get("validation_errors", []),
            }
            exit_code = 1
        else:
            try:
                response["execution"] = execute_response(
                    response,
                    output_dir=args.output_dir,
                    output_prefix=args.output_prefix,
                    regulation_file=args.regulation,
                    checkpoint_path=args.checkpoint,
                    device=args.device,
                    k=args.k,
                    top_m=args.top_m,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    llm_provider=args.llm_provider,
                    llm_api_key=args.llm_api_key,
                    llm_model=args.llm_model,
                )
            except Exception as exc:
                response["execution"] = {
                    "status": "failed",
                    "error": str(exc),
                }
                exit_code = 1

    print(json.dumps(response, indent=2))
    raise SystemExit(exit_code)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_resolution(args):
    resolution = {}
    if args.boundary:
        width, height = [float(part) for part in args.boundary.split(",")]
        resolution["boundary_size"] = (width, height)
    if args.area is not None:
        resolution["total_area"] = args.area
        resolution["area_unit"] = args.area_unit
    if args.entrance_point:
        x_val, y_val = [float(part) for part in args.entrance_point.split(",")]
        resolution["entrance_point"] = (x_val, y_val)
    return resolution or None


if __name__ == "__main__":
    main()
