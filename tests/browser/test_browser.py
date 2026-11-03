"""Real Gradio callbacks and browser PCM capture with fake inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright, expect


ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
    wav = Path(tmp) / "mic.wav"
    sf.write(wav, np.sin(np.arange(16000 * 30) * 2 * np.pi * 220 / 16000) * 0.3,
             16000, subtype="PCM_16")
    upload_wav = Path(tmp) / "upload.wav"
    sf.write(upload_wav, np.sin(np.arange(8000) * 2 * np.pi * 220 / 16000) * 0.3,
             16000, subtype="PCM_16")
    env = dict(os.environ, WEB_FAKE="true", WEB_PORT="7865", WEB_SHARE="false",
               WEB_HOST="127.0.0.1", PYTHONUTF8="1", GRADIO_ANALYTICS_ENABLED="false")
    with (Path(tmp) / "server.log").open("w+", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "app.py"],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(150):
                try:
                    if httpx.get("http://127.0.0.1:7865/config").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if server.poll() is not None:
                    raise AssertionError("server exited")
                time.sleep(0.2)
            else:
                raise AssertionError("server did not start")
            with sync_playwright() as p:
                browser = p.chromium.launch(channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None, args=[
                    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                    f"--use-file-for-fake-audio-capture={wav}",
                ])
                page = browser.new_page(permissions=["microphone"], accept_downloads=True)
                errors = []
                downloads = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("download", lambda d: downloads.append(d))
                page.goto("http://127.0.0.1:7865")
                page.wait_for_function("window.fillForm && window.fillForm.stats")
                page.wait_for_timeout(1200)

                def edit(label, value):
                    page.get_by_role("radio", name=label, exact=True).check()
                    page.wait_for_timeout(400)
                    page.get_by_role("textbox", name="Sửa giá trị trường đang chọn", exact=True).fill(value)
                    page.get_by_role("button", name="Lưu sửa tay", exact=True).click()
                    expect(page.get_by_role("textbox", name=label, exact=True)).to_have_value(value)

                edit("Họ tên", "Nguyễn Văn Nam")
                edit("Số điện thoại", "0901234567")
                edit("Địa chỉ", "Hà Nội")
                with page.expect_download() as download:
                    page.get_by_role("button", name="Tải JSON", exact=True).click()
                data = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
                assert data == {"full_name": "Nguyễn Văn Nam", "phone": "0901234567", "address": "Hà Nội"}
                edit("Số điện thoại", "")
                page.get_by_role("button", name="Tải JSON", exact=True).click()
                page.get_by_text("Số điện thoại cần 10 chữ số", exact=False).wait_for()
                page.wait_for_timeout(500)
                assert len(downloads) == 1, "failed validation downloaded stale JSON"
                edit("Số điện thoại", "0901234567")
                page.get_by_role("radio", name="Địa chỉ", exact=True).check()
                page.wait_for_timeout(400)
                page.get_by_role("button", name="Bật mic", exact=True).click()
                page.wait_for_function("window.fillForm.stats().captured > 8000")
                page.get_by_role("button", name="Tắt mic", exact=True).click()
                expect(page.get_by_role("textbox", name="Địa chỉ", exact=True)).to_have_value(
                    "Hà Nội âm thanh demo", timeout=15000)
                expect(page.get_by_role("textbox", name="Đoạn vừa nhận dạng", exact=True)).to_have_value("âm thanh demo")
                expect(page.get_by_role("button", name="Upload file âm thanh", exact=True)).to_be_enabled()
                page.locator('input[type="file"]').set_input_files(str(upload_wav))
                expect(page.get_by_role("textbox", name="Địa chỉ", exact=True)).to_have_value(
                    "Hà Nội âm thanh demo âm thanh demo", timeout=15000)
                expect(page.get_by_role("button", name="Upload file âm thanh", exact=True)).to_be_enabled()
                assert not errors, errors
                page.screenshot(path=str(Path(tmp) / "form-preview.png"), full_page=True)
                browser.close()
                print("PASS: manual edits, JSON download, microphone and uploaded WAV → VAD → fake ASR → selected field")
        except Exception:
            log.seek(0)
            print(log.read())
            raise
        finally:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
            else:
                server.terminate()
            server.wait(timeout=15)
