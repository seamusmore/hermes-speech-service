"""Bounded asynchronous SSE bridge with cooperative request cancellation."""
import asyncio
import queue
import threading
from contextlib import closing


async def stream_events(events, cancelled, on_close=lambda: None):
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    pending = queue.Queue(maxsize=4)
    finished = threading.Event()

    def notify():
        try:
            loop.call_soon_threadsafe(ready.set)
        except RuntimeError:
            # A disconnected consumer may already have closed its event loop.
            pass

    def put(item):
        while not cancelled.is_set():
            try:
                pending.put(item, timeout=.1)
                notify()
                return
            except queue.Full:
                pass

    def produce():
        try:
            with closing(events):
                for item in events:
                    if cancelled.is_set():
                        break
                    put(item)
        except Exception as exc:
            put(exc)
        finally:
            try:
                on_close()
            finally:
                finished.set()
                notify()

    thread = threading.Thread(target=produce, daemon=True, name='tts-stream')
    thread.start()
    try:
        while True:
            # Clear before checking the queue: a producer racing this check
            # schedules a new wakeup on this event loop.
            ready.clear()
            try:
                item = pending.get_nowait()
            except queue.Empty:
                if finished.is_set():
                    break
                await ready.wait()
                continue
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        cancelled.set()
        # The producer owns model cleanup and its runtime lease until it exits.

