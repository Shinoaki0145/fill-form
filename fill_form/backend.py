"""Audio capture and per-browser forms on one background event loop."""
import asyncio
import base64
from collections import deque
import threading
import time
import tomllib
import uuid
from pathlib import Path
import numpy as np
from .audio import AudioIngress, load_audio_file
from .config import ROOT
from .form import FormSession
from .runtime import AudioRunner
from .turns import TurnConfig

class ModelPool:
    def __init__(self, fake=False):
        self.fake = fake
        self.lock = threading.Lock()
        self.backend = None

    def load(self):
        with self.lock:
            if self.fake or self.backend:
                return
            from .asr import Gipformer

            folder = ROOT
            cfg = tomllib.loads((folder / "config.toml").read_text())["asr"]
            self.backend = Gipformer(
                folder / cfg["repo"],
                folder / cfg["icefall"],
                folder / cfg["model_dir"],
                "greedy_search",
                4,
            )

    def recognize(self, audio):
        with self.lock:
            if self.fake:
                return ("âm thanh demo", 0.01)
            return self.backend.recognize(audio)

    def new_vad(self):
        if self.fake:
            return lambda x: float(np.max(np.abs(x)) > 0.015)
        import torch
        from silero_vad import load_silero_vad

        model = load_silero_vad().to("cuda").eval()

        def vad(frame):
            with torch.inference_mode():
                return model(torch.from_numpy(frame).to("cuda"), 16000).item()

        return vad

class Client:
    def __init__(self, controller, vad):
        self.controller = controller
        self.events = deque(maxlen=200)
        self.mic = False
        self.packet_sequence = 1
        self.packets = {}
        self.last_seen = time.monotonic()
        self.closed = False
        self.ingress = AudioIngress()
        self.queue = asyncio.Queue()
        self.audio_queued = 0
        self.capture_error = None
        self.recording_id = None
        self.file_task = None
        self.file_target = None
        self.file_feeding = False
        self.session = FormSession(self)
        self.runner = AudioRunner(self.session, controller.pool, vad, TurnConfig(), controller.settings.silence)
        self.worker = asyncio.create_task(self.consume())

    def event(self, event):
        self.events.append(event)

    async def consume(self):
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(self.queue.get(), 0.032)
                except TimeoutError:
                    if not self.runner.busy:
                        continue
                    frame = None if self.mic or self.file_feeding else np.zeros(512, np.float32)
                if isinstance(frame, tuple):
                    callback, future = frame
                    if not future.cancelled():
                        try:
                            future.set_result(callback())
                        except Exception as exc:
                            future.set_exception(exc)
                    continue
                if not isinstance(frame, str):
                    self.audio_queued = max(0, self.audio_queued - 1)
                if isinstance(frame, str):
                    self.session.typed(frame)
                else:
                    await self.runner.consume(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.mic = False
            self.capture_error = (
                f"Audio pipeline lỗi: {type(e).__name__}. Tải lại trang."
            )
            self.event(
                dict(
                    event="audio_error",
                    generation=self.session.generation,
                    error=type(e).__name__,
                )
            )

    def enqueue(self, frames):
        if self.audio_queued + len(frames) > 96:
            self.mic = False
            self.capture_error = "Audio queue đầy: tắt/bật mic để thử lại"
            self.event(dict(event="audio_overflow", generation=self.session.generation))
            return
        for frame in frames:
            self.queue.put_nowait(frame)
            self.audio_queued += 1

    @property
    def file_active(self):
        return self.file_task is not None and not self.file_task.done()

    def start_file_stream(self, path, realtime=True):
        if self.mic or self.file_active or self.runner.busy or self.audio_queued:
            raise ValueError("Tắt mic và chờ audio hiện tại xử lý xong trước khi upload")
        if self.worker.done():
            raise ValueError("Audio pipeline đã dừng; tải lại trang")
        if not path or not Path(path).is_file():
            raise ValueError("Không tìm thấy file âm thanh")
        self.file_target = (self.session.selected, self.session.revisions[self.session.selected])
        self.session.error = ""
        self.capture_error = None
        self.file_feeding = True

        async def stream():
            try:
                frames, duration, rate = await asyncio.to_thread(load_audio_file, path)
                self.session.emit("file_stream_started", file=Path(path).name,
                                  audio_duration_s=round(duration, 3), sample_rate=rate)
                silence_frames = int(np.ceil((self.controller.settings.silence + 0.15) * 16000 / 512))
                deadline = time.perf_counter()
                for index, frame in enumerate(frames + [np.zeros(512, np.float32) for _ in range(silence_frames)]):
                    while self.audio_queued > 64:
                        if self.worker.done():
                            raise ValueError("Audio pipeline đã dừng; tải lại trang")
                        await asyncio.sleep(0.016)
                    self.queue.put_nowait(frame)
                    self.audio_queued += 1
                    if realtime and index < len(frames):
                        deadline += 512 / 16000
                        await asyncio.sleep(max(0, deadline - time.perf_counter()))
                    else:
                        await asyncio.sleep(0)
                self.file_feeding = False
                while self.audio_queued or self.runner.busy:
                    if self.worker.done():
                        raise ValueError("Audio pipeline đã dừng; tải lại trang")
                    await asyncio.sleep(0.032)
                self.session.emit("file_stream_finished", file=Path(path).name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.session.error = f"Không xử lý được file: {exc}"
                self.session.emit("file_error", error_type=type(exc).__name__)
            finally:
                self.file_feeding = False
                self.file_target = None

        self.file_task = asyncio.create_task(stream())

    async def close(self):
        self.closed = True
        self.mic = False
        if self.file_active:
            self.file_task.cancel()
            await asyncio.gather(self.file_task, return_exceptions=True)
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)
        await self.runner.close()


class Controller:
    def __init__(self, settings, pool=None):
        self.settings = settings
        self.pool = pool or ModelPool(settings.fake)
        self.pool.load()
        self.clients = {}
        self.opening = 0
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.call(self.reaper_start())

    def call(self, coro, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    async def reaper_start(self):
        self.reaper = asyncio.create_task(self.expire())

    async def expire(self):
        while True:
            await asyncio.sleep(10)
            for sid, c in list(self.clients.items()):
                if time.monotonic() - c.last_seen > 120:
                    try:
                        await c.close()
                    except Exception:
                        pass
                    self.clients.pop(sid, None)

    def close(self):
        async def cleanup():
            self.reaper.cancel()
            await asyncio.gather(self.reaper, return_exceptions=True)
            await asyncio.gather(
                *(c.close() for c in list(self.clients.values())),
                return_exceptions=True,
            )
            self.clients.clear()

        self.call(cleanup())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()

    def dispatch(self, sid, action, *args):
        async def run():
            operation = action
            arguments = args
            if operation == "open":
                if len(self.clients) + self.opening >= 2:
                    raise ValueError("Đã đủ 2 phiên, đóng tab cũ hoặc đợi 120s")
                self.opening += 1
                try:
                    vad = await asyncio.to_thread(self.pool.new_vad)
                    key = uuid.uuid4().hex
                    self.clients[key] = Client(self, vad)
                    return key
                finally:
                    self.opening -= 1
            client = self.clients[sid]
            client.last_seen = time.monotonic()
            methods = {"select_field": client.session.select,
                       "edit_field": client.session.edit,
                       "export_form": client.session.export,
                       "audio_file": client.start_file_stream}
            control = None
            if operation == "form_packet" or operation in methods:
                if client.closed or client.worker.done():
                    raise ValueError("Phiên đã dừng; tải lại trang")
                if operation == "form_packet":
                    packet, command, *parameters = arguments
                else:
                    command, parameters = operation, arguments
                if command not in methods:
                    raise ValueError("Thao tác form không hợp lệ")
                future = self.loop.create_future()
                control = (lambda: methods[command](*parameters), future)
                if operation == "form_packet":
                    operation, arguments = "packet", ({**packet, "_control": control},)
                else:
                    client.queue.put_nowait(control)
                    return await asyncio.wait_for(future, 5)
            if operation == "packet":
                packet = arguments[0]
                seq = packet["sequence"]
                if type(seq) is not int or seq < 1:
                    raise ValueError("Sequence không hợp lệ")
                if seq < client.packet_sequence:
                    if control:
                        raise ValueError("Gói điều khiển đã cũ; thử lại")
                    return
                if seq > client.packet_sequence + 32:
                    raise ValueError("Audio transport mất thứ tự quá lâu, tải lại trang")
                client.packets[seq] = packet
                while client.packet_sequence in client.packets:
                    packet = client.packets.pop(client.packet_sequence)
                    client.packet_sequence += 1
                    if packet["mic"] and packet.get("recording_id") != client.recording_id:
                        client.recording_id = packet.get("recording_id")
                        client.capture_error = None
                        if client.file_active:
                            client.capture_error = "Đang xử lý file; chờ xong rồi bật mic"
                        elif any(c.mic for c in self.clients.values() if c is not client):
                            client.capture_error = "Một phiên khác đang dùng mic"
                        else:
                            client.ingress = AudioIngress()
                            client.mic = True
                    if packet.get("audio") and client.mic:
                        audio = np.frombuffer(base64.b64decode(packet["audio"], validate=True), dtype="<i2")
                        client.enqueue(client.ingress.push(packet["rate"], audio))
                    if not packet["mic"] and client.mic:
                        client.enqueue(client.ingress.finish())
                        client.mic = False
                    queued_control = packet.get("_control")
                    if isinstance(queued_control, tuple) and len(queued_control) == 2 and callable(queued_control[0]) and isinstance(queued_control[1], asyncio.Future):
                        client.queue.put_nowait(queued_control)
                if control:
                    return await asyncio.wait_for(control[1], 5)
            elif operation == "text":
                client.queue.put_nowait(arguments[0])
            elif operation == "mic":
                enabled = arguments[0]
                if enabled and client.file_active:
                    raise ValueError("Đang xử lý file; chờ xong rồi bật mic")
                if enabled and any(c.mic for c in self.clients.values() if c is not client):
                    raise ValueError("Một phiên khác đang dùng mic")
                if enabled and not client.mic:
                    client.ingress = AudioIngress()
                if not enabled and client.mic:
                    client.enqueue(client.ingress.finish())
                client.mic = enabled
            elif operation == "audio":
                if client.mic and arguments[0] is not None:
                    client.enqueue(client.ingress.push(*arguments[0]))
            elif operation == "snapshot":
                session = client.session
                return dict(mic=client.mic, capture_error=client.capture_error,
                            file_active=client.file_active,
                            events=list(client.events), asr_busy=client.runner.busy,
                            form=dict(values=dict(session.values), selected=session.selected,
                                      transcript=session.transcript, error=session.error))
            elif operation == "close":
                await client.close()
                self.clients.pop(sid, None)
            else:
                raise ValueError("Thao tác không hợp lệ")

        return self.call(run(), timeout=60 if action == "open" else 10)
