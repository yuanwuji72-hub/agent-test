import time
import os
from openai import OpenAI


class ConversationManager:
    """对话管理器"""

    def __init__(self, max_context_length=4000):
        self.max_context_length = max_context_length  # 最大token数
        self.history = []  # 完整的对话历史

    @staticmethod
    def estimate_tokens_static(text: str) -> int:
        if not text: return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        is_cn = chinese_chars / max(len(text), 1) > 0.3
        if is_cn:
            return max(1, int(len(text) / 1.5))
        return max(1, int(len(text.split()) * 1.3))

    def add_exchange(self, user_input:str, assistant_response:str):
        """添加一轮对话"""
        self.history.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": time.time()
        })

    def build_context(self, current_query:str, include_history:bool =True)->list[dict]:
        """构建当前查询的上下文"""

        if not include_history or not self.history:
            # 没有历史或不需要历史，只返回当前查询
            return [{"role": "user", "content": current_query}]

        # 从最近的对话开始，逐步添加历史，直到达到长度限制
        context = []
        context_length = self.estimate_tokens_static(current_query)

        # 添加当前查询
        context.insert(0, {"role": "user", "content": current_query})

        # 从最近到最远添加历史
        for exchange in reversed(self.history):
            user_tokens = self.estimate_tokens_static(exchange["user"])
            assistant_tokens = self.estimate_tokens_static(exchange["assistant"])

            # 检查是否会超出限制
            if context_length + user_tokens + assistant_tokens > self.max_context_length:
                break

            # 添加助理回复（在用户输入之前）
            context.insert(0, {"role": "assistant", "content": exchange["assistant"]})
            context.insert(0, {"role": "user", "content": exchange["user"]})

            context_length += user_tokens + assistant_tokens

        return context



class LLM:
    def __init__(self,api_key:str,base_url:str,model:str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    def invoke(self,message:list[dict])->str:
        try:
            _llm = self.llm.chat.completions.create(
                model=self.model,
                messages=message
            )
            return _llm.choices[0].message.content
        except Exception as e:
            return f"[LLM Error] {type(e).__name__}: {e}"





class Agent:
    def __init__(self,llm:LLM,system_prompt:str,max_token:int):
        self.llm = llm
        self.system_prompt = system_prompt
        system_tokens = ConversationManager.estimate_tokens_static(system_prompt)
        effective_max = max(0, max_token - system_tokens)
        self.conversation_manager = ConversationManager(effective_max)

    def chat(self,user_input:str)->str:
        messages = [{"role":"system","content":self.system_prompt}]
        messages.extend(self.conversation_manager.build_context(user_input))
        response = self.llm.invoke(messages)
        if not response.startswith("[LLM Error]"):
            self.conversation_manager.add_exchange(user_input,response)
        return response

if "__main__" == __name__:
    models = LLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("qw_url"),
        model="qwen3.7-flash-2026-07-15"
    )
    system_prompts = "你是一位高冷的爱助手，以简洁高效著称，按照这个人设回答用户问题"
    agent = Agent(models,system_prompts,4000)
    while True:
        user_inputs = input("请输入你的问题")
        result = agent.chat(user_inputs)
        print(result)