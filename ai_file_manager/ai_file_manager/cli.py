"""
cli.py — Command-line interface for AI File Manager.
Can be used directly or triggered from Windows Explorer right-click.

Usage:
  python cli.py sort "C:\\Users\\Dash\\Documents\\MyFolder" --mode topic
  python cli.py sort "C:\\Users\\Dash\\Documents\\MyFolder" --mode date --execute
  python cli.py undo "C:\\Users\\Dash\\Documents\\MyFolder"
  python cli.py scan "C:\\Users\\Dash\\Documents\\MyFolder"
  python cli.py server
"""

from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path


def cmd_scan(args):
    from core.scanner import FolderScanner
    scanner = FolderScanner(recursive=not args.flat)
    manifest = scanner.scan(args.folder)
    print(manifest.summary())
    print("\nFiles by category:")
    for cat, files in manifest.by_category().items():
        print(f"  {cat:12s} : {len(files)}")


def cmd_sort(args):
    """Full sort pipeline — runs via asyncio."""
    asyncio.run(_async_sort(args))


async def _async_sort(args):
    from core.scanner import FolderScanner
    from core.router import FileRouter
    from core.organizer import FileOrganizer
    from models.llm_client import LLMClient
    from models.embedding_client import SemanticClusterer

    folder = Path(args.folder).resolve()
    mode = args.mode
    execute = args.execute

    print(f"\n{'='*60}")
    print(f"  AI File Manager — Sort Mode: {mode.upper()}")
    print(f"  Folder: {folder}")
    print(f"{'='*60}\n")

    # ── Step 1: Scan ─────────────────────────────────────────────────────────
    print("📂  [1/5] Scanning files...")
    scanner = FolderScanner(recursive=not args.flat)
    manifest = scanner.scan(folder)
    print(f"     Found {manifest.count} files  ({manifest.total_size_bytes // 1024 // 1024} MB)\n")

    if manifest.count == 0:
        print("     No files found. Exiting.")
        return

    # ── Step 2: Process through pipelines ────────────────────────────────────
    print("🔍  [2/5] Extracting content from files...")
    router = FileRouter()
    results = await router.process_many(manifest.files, max_concurrent=3)
    ok = sum(1 for r in results if r.success)
    print(f"     Processed: {ok}/{len(results)} successful\n")

    # ── Step 3: LLM metadata extraction ──────────────────────────────────────
    print("🧠  [3/5] Running LLM classification (this may take a while)...")
    llm = LLMClient.get_instance()
    enriched: list[dict] = []
    for i, result in enumerate(results, 1):
        fi = result.file_info
        text = result.text or f"File: {fi['name']} (type: {fi['category']})"
        meta = llm.extract_metadata(text, fi["name"], fi["category"])
        enriched.append({"file_info": fi, "text": text, "metadata": meta})

        # Progress indicator
        if i % 5 == 0 or i == len(results):
            pct = int(i / len(results) * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r     [{bar}] {pct}%  ({i}/{len(results)})", end="", flush=True)

    print("\n     LLM classification complete\n")

    # ── Step 4: Cluster (relation mode only) ─────────────────────────────────
    if mode == "relation":
        print("🔗  [4/5] Computing semantic clusters...")
        clusterer = SemanticClusterer()
        texts = [e["text"] or e["file_info"]["name"] for e in enriched]
        labels = clusterer.cluster_texts(texts)

        cluster_topics: dict[str, list] = {}
        for label, entry in zip(labels, enriched):
            cluster_topics.setdefault(label, []).append(
                entry["metadata"].get("topic", "")
            )

        cluster_names = {
            label: llm.name_cluster(topics)
            for label, topics in cluster_topics.items()
        }

        for label, entry in zip(labels, enriched):
            entry["cluster_label"] = cluster_names.get(label, label)

        n_clusters = len(cluster_names)
        print(f"     Found {n_clusters} clusters\n")
    else:
        print(f"ℹ️   [4/5] Skipping clustering (mode={mode})\n")
        for entry in enriched:
            entry["cluster_label"] = ""

    # ── Step 5: Build plan ────────────────────────────────────────────────────
    print("📋  [5/5] Building folder plan...")
    organizer = FileOrganizer(folder)
    plan = organizer.build_plan(enriched, mode=mode)

    print()
    print(plan.preview())
    print()

    # ── Execute or dry-run ────────────────────────────────────────────────────
    if not execute:
        print("ℹ️   Dry-run mode. No files were moved.")
        print("     Add --execute flag to apply the sort.\n")
        return

    # Confirm before moving
    if not args.yes:
        answer = input(f"⚠️   Move {len(plan.entries)} files? [y/N]: ").strip().lower()
        if answer != "y":
            print("     Aborted.\n")
            return

    print("🚀  Executing sort...")
    result = organizer.execute_plan(plan)

    print(f"\n{'='*60}")
    if result["success"]:
        print(f"✅  Done! Moved {result['moved']} files.")
        print(f"    Undo session: {result.get('undo_session', 'N/A')}")
    else:
        print(f"⚠️   Completed with errors. Moved {result['moved']} files.")
        for err in result["errors"]:
            print(f"    ✗ {err}")
    print(f"{'='*60}\n")


def cmd_undo(args):
    from core.organizer import FileOrganizer
    folder = Path(args.folder).resolve()
    organizer = FileOrganizer(folder)

    if args.list:
        sessions = organizer.list_undo_sessions()
        if not sessions:
            print("No undo sessions found.")
            return
        print(f"\n{'─'*50}")
        print(f"  Undo sessions for: {folder.name}")
        print(f"{'─'*50}")
        for s in sessions:
            print(f"  {s['session_id']}  |  {s['created_at']}  |  {s['move_count']} moves")
        print()
        return

    session_id = getattr(args, "session_id", None)
    if session_id:
        from core.undo_manager import UndoManager
        um = UndoManager(folder)
        result = um.undo_session(session_id)
    else:
        result = organizer.undo_last()

    if result["success"]:
        print(f"✅  Undo successful: {result['restored']} files restored.")
    else:
        print(f"⚠️   Undo partial: {result['restored']} restored.")
        for err in result.get("errors", []):
            print(f"    ✗ {err}")


def cmd_server(args):
    import uvicorn
    from utils.config import load_config
    cfg = load_config()
    print(f"\n🌐  Starting AI File Manager API server...")
    print(f"    http://{cfg.api.host}:{cfg.api.port}\n")
    uvicorn.run(
        "api.server:app",
        host=cfg.api.host,
        port=int(cfg.api.port),
        workers=int(cfg.api.workers),
        reload=False,
    )


# ─── Argument Parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-file-manager",
        description="AI-powered file organizer — runs locally on low-VRAM GPU",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── scan ──────────────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Scan a folder and show file statistics")
    p_scan.add_argument("folder", help="Path to folder to scan")
    p_scan.add_argument("--flat", action="store_true", help="Non-recursive scan")

    # ── sort ──────────────────────────────────────────────────────────────────
    p_sort = sub.add_parser("sort", help="Sort files in a folder using AI")
    p_sort.add_argument("folder", help="Path to folder to sort")
    p_sort.add_argument(
        "--mode",
        choices=["topic", "date", "relation"],
        default="topic",
        help="Sorting strategy: topic | date | relation (default: topic)",
    )
    p_sort.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default: dry-run / preview only)",
    )
    p_sort.add_argument(
        "--flat",
        action="store_true",
        help="Only process files in the top-level folder",
    )
    p_sort.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # ── undo ──────────────────────────────────────────────────────────────────
    p_undo = sub.add_parser("undo", help="Undo the last sort operation")
    p_undo.add_argument("folder", help="Path to the sorted folder")
    p_undo.add_argument("--session-id", dest="session_id", help="Specific session to undo")
    p_undo.add_argument("--list", action="store_true", help="List available undo sessions")

    # ── server ────────────────────────────────────────────────────────────────
    p_server = sub.add_parser("server", help="Start the FastAPI server")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "scan":   cmd_scan,
        "sort":   cmd_sort,
        "undo":   cmd_undo,
        "server": cmd_server,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
