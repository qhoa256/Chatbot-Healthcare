# CHATBOT TƯ VẤN SỨC KHỎE TIẾNG VIỆT

## Tổng quan
Dự án xây dựng chatbot trả lời tự động về các vấn đề sức khỏe bằng tiếng Việt, dựa trên mô hình LSTM (Long Short-Term Memory). Chatbot được huấn luyện trên bộ dữ liệu từ trang web Alobacsi.com, nơi người dùng đặt câu hỏi và nhận được câu trả lời từ các bác sĩ.

## Phương pháp huấn luyện

### 1. Thu thập và tiền xử lý dữ liệu
- **Nguồn dữ liệu**: Dữ liệu được thu thập từ trang web Alobacsi.com bao gồm các cặp câu hỏi-trả lời về chủ đề sức khỏe.
- **Tiền xử lý văn bản tiếng Việt**:
  - Chuẩn hóa Unicode
  - Loại bỏ biểu tượng cảm xúc, ký tự đặc biệt, URL
  - Chuyển văn bản về chữ thường
  - Loại bỏ số, dấu câu
  - Phân đoạn từ tiếng Việt sử dụng thư viện `pyvi.ViTokenizer`
  - Thêm các token đặc biệt `<START>` và `<END>` vào câu trả lời để đánh dấu bắt đầu và kết thúc câu

### 2. Xây dựng từ điển và mã hóa dữ liệu
- Tạo từ điển riêng biệt cho đầu vào (câu hỏi) và đầu ra (câu trả lời)
- Chuyển đổi văn bản thành dạng one-hot encoding cho mô hình LSTM
- Xác định chiều dài tối đa cho câu hỏi và câu trả lời để chuẩn hóa kích thước đầu vào

### 3. Kiến trúc mô hình LSTM Encoder-Decoder
Mô hình sử dụng kiến trúc Sequence-to-Sequence với cơ chế Encoder-Decoder:

- **Encoder**:
  - Input layer với kích thước (None, num_encoder_tokens)
  - LSTM layer với 256 đơn vị, trả về state
  - Encoder chuyển đổi câu hỏi đầu vào thành các vector trạng thái ẩn

- **Decoder**:
  - Input layer với kích thước (None, num_decoder_tokens)
  - LSTM layer với 256 đơn vị, nhận trạng thái từ encoder
  - Dense layer với hàm kích hoạt softmax để tạo ra phân phối xác suất cho từng token đầu ra

### 4. Quá trình huấn luyện
- **Kích thước batch**: 10
- **Số epochs**: 200
- **Dimensionality**: 256
- **Hàm mất mát**: categorical_crossentropy
- **Optimizer**: Adam
- **Quá trình huấn luyện**: Mô hình được huấn luyện để tối thiểu hóa sự khác biệt giữa câu trả lời dự đoán và câu trả lời thực tế trong tập dữ liệu

## Kỹ thuật sử dụng trong dự án

### 1. Xử lý ngôn ngữ tự nhiên tiếng Việt
- Sử dụng `pyvi` để phân đoạn từ tiếng Việt
- Kỹ thuật chuẩn hóa dấu câu và dấu thanh tiếng Việt
- Loại bỏ stopwords tiếng Việt

### 2. Kiến trúc Sequence-to-Sequence (Seq2Seq)
- Mô hình Encoder-Decoder LSTM để xử lý chuỗi văn bản đầu vào và tạo ra chuỗi văn bản đầu ra
- Sử dụng cơ chế trạng thái ẩn để truyền thông tin từ encoder sang decoder

### 3. Quá trình sinh câu trả lời
- **Inference Mode**: Trong quá trình dự đoán, encoder tạo ra vector trạng thái từ câu hỏi
- **Beam Search**: Kỹ thuật tìm kiếm để tạo ra câu trả lời có xác suất cao nhất
- **Tokenization và De-tokenization**: Chuyển đổi qua lại giữa văn bản và vector để xử lý và sinh câu trả lời

## Hướng dẫn sử dụng

### Cài đặt môi trường ảo

#### Sử dụng venv (Khuyến nghị)
```bash
# Tạo môi trường ảo
python -m venv healthcare_chatbot

# Kích hoạt môi trường ảo (Windows)
healthcare_chatbot_env\Scripts\activate

# Kích hoạt môi trường ảo (Linux/Mac)
source healthcare_chatbot_env/bin/activate
```

#### Sử dụng Conda
```bash
# Tạo môi trường Conda
conda create --name healthcare_chatbot_env python=3.8

# Kích hoạt môi trường
conda activate healthcare_chatbot_env
```

### Cài đặt các thư viện cần thiết
```bash
# Cài đặt các thư viện từ tệp requirements.txt
pip install -r requirements.txt
```

### Chạy ứng dụng Chatbot
```bash
# Chạy ứng dụng Flask
python app.py
```
Sau khi chạy, bạn có thể truy cập chatbot qua địa chỉ: http://localhost:5000

### Kết thúc phiên làm việc
```bash
# Để thoát môi trường ảo venv
deactivate

# Hoặc để thoát môi trường Conda
conda deactivate
```

## Cấu trúc dự án
- `model/`: Thư mục chứa các mô-đun của mô hình
  - `chatbot.py`: Định nghĩa lớp ChatBot xử lý tương tác với người dùng
  - `prediction.py`: Mô-đun dự đoán câu trả lời từ câu hỏi đầu vào
  - `preprocessing.py`: Xử lý dữ liệu và chuẩn bị cho mô hình
  - `utils.py`: Các hàm tiện ích cho xử lý văn bản tiếng Việt
- `data/`: Thư mục chứa dữ liệu huấn luyện
- `templates/`: Giao diện người dùng
- `app.py`: Ứng dụng Flask để chạy chatbot trên web
- `Training_model.ipynb`: Notebook huấn luyện mô hình LSTM

## Lưu ý
- Mô hình đã được huấn luyện trên bộ dữ liệu hạn chế, nên câu trả lời chỉ mang tính chất tham khảo
- Không nên sử dụng chatbot này để thay thế tư vấn y tế chuyên nghiệp
- Khi gặp vấn đề sức khỏe nghiêm trọng, hãy luôn tham khảo ý kiến của bác sĩ
