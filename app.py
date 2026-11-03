"""Launch the standalone voice form."""
import atexit
from pathlib import Path
from fill_form.config import Settings, ROOT
from fill_form.backend import Controller
from fill_form.ui import build_ui


def main():
    settings = Settings.load()
    controller = Controller(settings)
    atexit.register(controller.close)
    ui = build_ui(controller, settings)
    ui.queue(max_size=64).launch(
        server_name=settings.host, server_port=settings.port, share=settings.share,
        auth=None,
        max_file_size="50mb",
        show_error=False, allowed_paths=[], blocked_paths=[str(ROOT)],
        js=Path(__file__).with_name("fill_form").joinpath("browser_audio.js").read_text(encoding="utf-8"),
    )


if __name__ == "__main__":
    main()
