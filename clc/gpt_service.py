import os
from typing import Dict, Union, Optional
from typing import List

from accelerate import load_checkpoint_and_dispatch
from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer


class ChatGLMService(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p: float = 0.9
    history: List = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response

    def load_model(self,
                   model_name_or_path: str = "THUDM/chatglm-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        # 加载模型到CPU
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = self.model.eval()  # 设置为评估模式

    def auto_configure_device_map(self, num_gpus: int) -> Dict[str, int]:
        # 由于现在是CPU，因此不需要GPU配置
        device_map = {'transformer.word_embeddings': 'cpu',
                      'transformer.final_layernorm': 'cpu', 'lm_head': 'cpu'}
        return device_map

    def load_model_on_gpus(self, model_name_or_path: Union[str, os.PathLike], num_gpus: int = 1,
                           multi_gpu_model_cache_dir: Union[str, os.PathLike] = "./temp_model_dir",
                           ):
        # 在CPU上加载模型，不涉及GPU配置
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = self.model.eval()  # 设置为评估模式

        # CPU 上不需要配置 device_map，直接加载即可
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        print(f"Model loaded successfully on CPU.")
