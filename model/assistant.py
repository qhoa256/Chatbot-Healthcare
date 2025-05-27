from google.generativeai import configure, GenerativeModel
import os
from dotenv import load_dotenv
import re

load_dotenv()

prompt_assistant_cfg = {
    "model_type": "gemini-2.0-flash",
    "api_key": os.getenv("GEMINI_API_KEY", "AIzaSyAxcgWc9ahf3XfMs48cj7NetddiumCTG6E")  
}

model_type = prompt_assistant_cfg["model_type"]
api_key = prompt_assistant_cfg["api_key"]

CHATBOT_PROMPT = '''Vai trò: Bạn là một chatbot tư vấn sức khỏe chuyên nghiệp, cung cấp thông tin y tế đáng tin cậy.

Nội dung câu hỏi: "{text}"

Yêu cầu phản hồi:
- Trả lời CỰC KỲ ngắn gọn, đơn giản, đúng trọng tâm câu hỏi
- Cung cấp thông tin y khoa chính xác và cập nhật
- Đưa ra chẩn đoán cụ thể.
- Tránh sử dụng thuật ngữ y khoa phức tạp, ưu tiên ngôn ngữ dễ hiểu
- Không trả lời thừa thông tin không liên quan đến câu hỏi
- Sử dụng định dạng ngắn gọn, rõ ràng
- Không cung cấp các thông tin thừa như tư vấn đến bác sĩ, đường dẫn đến trang web khác,...

Hướng dẫn phong cách:
- Giọng điệu chuyên nghiệp nhưng thân thiện
- Thể hiện sự đồng cảm khi phù hợp
- Sử dụng ngôn ngữ tích cực và khuyến khích
- Không gây hoang mang hay lo lắng không cần thiết

QUAN TRỌNG: Phản hồi phải bằng tiếng Việt, ngắn gọn và đúng trọng tâm'''

class PromptAssistant:
    def __init__(self):
        self.model = GenerativeModel(model_type.lower())
        self.cfg_model = configure(api_key=api_key)
    
    def clean_markdown(self, text):
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'^\s*\*\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^#+\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _send_to_model(self, prompt):
        response = self.model.generate_content(
            prompt,
            generation_config={
                "top_k": 32,
                "top_p": 0.95,
                "temperature": 0.7,
                "max_output_tokens": 2048
            }
        )
        return self.clean_markdown(response.text)
    
    def reply(self, text):
        prompt = CHATBOT_PROMPT.format(text=text)
        return self._send_to_model(prompt)