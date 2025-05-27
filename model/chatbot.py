import re
import numpy as np
from .prediction import decode_response
from .preprocessing import clean_data
from .assistant import PromptAssistant

chatbot = PromptAssistant()

class ChatBot:
  def __init__(self, max_encoder_seq_length, num_encoder_tokens, input_features_dict, decode_response):
    self.max_encoder_seq_length = max_encoder_seq_length
    self.num_encoder_tokens = num_encoder_tokens
    self.input_features_dict = input_features_dict
    self.decode_response = decode_response
    self.negative_responses = ("không", "no", "cảm ơn", "sai", "xin lỗi", "sorry")
    self.exit_commands = ("tạm biệt", "dừng lại", "thoát", "goodbye", "bye")

  def start_chat(self, user_response):
    if user_response in self.negative_responses:
      return "Cảm ơn và hẹn gặp lại"
    return self.chat(user_response)

  def chat(self, reply):
    if reply in self.exit_commands:
      return "Cảm ơn và hẹn gặp lại"
    return self.generate_response(clean_data(reply))

  def text_to_matrix(self, user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
      (1, self.max_encoder_seq_length, self.num_encoder_tokens),
      dtype='float32')
    for timestep, token in enumerate(tokens):
      if token in self.input_features_dict:
        user_input_matrix[0, timestep, self.input_features_dict[token]] = 1.
    return user_input_matrix

  def generate_response(self, user_input):
    input_matrix = self.text_to_matrix(user_input)
    response = self.decode_response(input_matrix)
    response = response.replace("_",' ')
    response = response.replace("<START>",'')
    response = response.replace("<END>",'')
    return chatbot.reply(user_input)

