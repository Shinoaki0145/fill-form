# Fill Form — ASR tiếng Việt

Project độc lập: microphone → Silero VAD → Gipformer → trường đã chọn.
Chỉ chứa code phục vụ nhận dạng và điền form; không cần project hội thoại,
LLM, TTS, tài khoản đăng nhập hoặc upload ZIP.

## Cấu trúc

```text
fill-form/
├── app.py                       # Giao diện Gradio
├── fill_form/
│   ├── audio.py                 # Resample PCM về 16 kHz
│   ├── browser_audio.js         # Thu microphone trong trình duyệt
│   ├── asr.py                   # Gipformer CUDA
│   ├── setup.py                 # Tải model và source upstream đã ghim
│   ├── turns.py                 # Buffer audio và khoảng nghỉ
│   ├── runtime.py               # Chốt đoạn và chạy ASR tuần tự
│   ├── backend.py               # Phiên, hàng đợi audio và điều khiển form
│   ├── form.py                  # Dữ liệu, chuẩn hóa và kiểm tra form
│   ├── config.py                # Cấu hình môi trường
│   └── ui.py                    # Chọn trường, sửa tay, tải JSON
├── config.toml
├── pyproject.toml
├── uv.lock
├── Run_Fill_Form_Colab.ipynb
└── tests/
```

## Chạy trên Colab từ GitHub

1. Đẩy cả thư mục `fill-form` vào gốc repo ASR hiện tại, hoặc đẩy nội dung bên
   trong thư mục này vào gốc một repo riêng.
2. Mở `Run_Fill_Form_Colab.ipynb`, chọn **T4 GPU**.
3. Đặt `REPO_URL` và `REPO_BRANCH` theo repo/nhánh bạn vừa đẩy. Mặc định dùng
   `https://github.com/Shinoaki0145/ASR.git` và nhánh mặc định.
4. Chạy đến cell hiện link Gradio, mở link và cấp quyền mic. Notebook nhận cả
   cấu trúc repo độc lập và cấu trúc có thư mục `fill-form` ở gốc repo.
5. Chỉ chạy cell `stop_server()` khi sử dụng xong.

Notebook tái sử dụng source có sẵn, không tự pull hoặc ghi đè checkout. Nếu
checkout cũ chưa có code mới, cập nhật nó hoặc dùng runtime Colab mới.
Model và vendor được tải vào chính project; không cần thư mục source bên ngoài.

## Sử dụng

Chọn họ tên, số điện thoại hoặc địa chỉ → bật mic → nói → nghỉ khoảng 0,7 giây.
Các đoạn nói nối vào trường được chọn khi đoạn bắt đầu. Số điện thoại đọc từng
chữ số; không hỗ trợ cách đọc nhóm như “chín mươi”.

Hoặc tắt mic, chờ audio cũ xử lý xong, chọn trường rồi bấm **Upload file âm thanh**.
Hỗ trợ WAV, FLAC, MP3 qua SoundFile; các định dạng khác như M4A cần `ffmpeg`
(notebook Colab tự cài nếu thiếu). Giới hạn 50 MB và 5 phút mỗi file.
Audio được resample về 16 kHz và đẩy qua Silero VAD theo thời gian thực như mic.
Các đoạn trong cùng file điền vào trường được chọn lúc bắt đầu upload; đổi
trường khi đang xử lý không chuyển đích file. Sửa tay vẫn được ưu tiên.
Không nhận mic hoặc file mới trong khi file hiện tại chưa xử lý xong. Chờ trạng
thái sẵn sàng trước khi tải JSON. File hỏng hoặc chỉ có im lặng không xóa dữ liệu cũ.

Muốn sửa/xóa: chọn trường, nhập toàn bộ giá trị mới vào ô sửa rồi bấm **Lưu sửa
tay**; để trống để xóa. ASR đến muộn không ghi đè bản sửa đã lưu. Bản gõ chưa
lưu không thuộc form; đổi trường sẽ nạp lại ô sửa theo trường mới.

Tắt mic và chờ nhận dạng xong, kiểm tra các trường rồi **Tải JSON**. Form mẫu
yêu cầu họ tên, địa chỉ và số điện thoại `0` + 9 số hoặc `+84` + 9 số.
Form không tự submit; giá trị form chỉ nằm trong RAM của phiên. File upload
được Gradio lưu tạm trên server; Gradio dọn mỗi 10 phút các file tạm cũ hơn 1 giờ.
Tải JSON trước khi đóng tab/dừng server.
Hai phiên độc lập, một mic hoạt động cùng lúc.
Link public không có đăng nhập; ai có link đều có thể dùng ứng dụng.

## Chạy trên Linux/CUDA

Từ thư mục project, dùng Python 3.12 và uv:

```bash
uv sync --locked --extra cuda
uv run --locked --extra cuda python -m fill_form.setup --config config.toml
uv run --locked --extra cuda python app.py
```

Cài `ffmpeg` trên máy Linux nếu cần đọc thêm các định dạng audio ngoài SoundFile.

Mặc định mở localhost:7860. Sao chép `.env.example` thành `.env` nếu muốn đổi
port, ngưỡng im lặng hoặc bật share link. Dùng `ASR_ENV_PATH` để chọn file khác.

## Demo và kiểm tra không GPU

```bash
uv sync --locked --extra dev
WEB_FAKE=true uv run --locked python app.py
uv run --locked python -X utf8 -m unittest discover -s tests -v
uv run --locked playwright install chromium
uv run --locked python tests/browser/test_browser.py
```

Demo dùng VAD theo âm lượng và transcript cố định, không đánh giá chất lượng
model thật. Runtime CUDA và lockfile nhắm Linux x86_64/Colab. GPU thật, tên riêng,
chuỗi số và độ trễ cần đo trên thiết bị thực tế.
