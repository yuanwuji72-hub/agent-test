from typing import Generator

from openai import OpenAI
import os
class LLM:
    def __init__(self,api_key:str,base_url:str,model:str):
        if not api_key or not base_url or not model:
            raise ValueError("api_key , base_url ,model 不能为空，请检查环境变量配置")
        self.model = model
        self.llm = OpenAI(
            api_key = api_key,
            base_url = base_url,
        )
    def stream(self,user_input: str = None, messages: list[dict] = None)-> Generator[str, None, None]:
        try:
            if messages is None:
                if user_input is None:
                    raise ValueError("user_input 和 messages 至少提供一个")
                messages = [{"role": "user", "content": user_input}]
            result = self.llm.chat.completions.create(
            model = self.model,
            messages=messages,
            stream=True
            )

            for item in result:
                if item.choices and item.choices[0].delta.content:
                    yield item.choices[0].delta.content
        except Exception as e:
            yield f"\n[ERROR] 流式请求失败: {e}"

if __name__ == "__main__":
    llm = LLM(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("qw_url"),
            model = "qwen3.7-flash-2026-07-15"
    )
    while True:
        user_input = input("你")
        res = llm.stream(user_input)
        for i in res:
            print(i,end="",flush=True)
            print()