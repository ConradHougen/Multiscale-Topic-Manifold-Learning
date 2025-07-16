#!/usr/bin/env python3
"""
Development Watcher for MSTML - Auto rebuild on .pyx change
"""

import time
import subprocess
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_FILES = [
    os.path.abspath("mstml/fast_encode_tree.pyx"),
    # Add more .pyx files if needed
]

class RebuildHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path in WATCH_FILES:
            print(f"\nDetected change: {event.src_path}")
            rebuild()

def rebuild():
    try:
        print("Rebuilding Cython + editable package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("Rebuild complete.")
    except subprocess.CalledProcessError as e:
        print("Build failed:", e.stderr)

def main():
    print("Watching for changes in .pyx files...")
    observer = Observer()
    handler = RebuildHandler()
    for path in set(os.path.dirname(f) for f in WATCH_FILES):
        observer.schedule(handler, path=path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
