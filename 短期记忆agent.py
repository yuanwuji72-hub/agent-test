import os
from collections import deque
from openai import AsyncOpenAI
import asyncio

class LLM:
    def __init__(self,api_key:str,base_url:str,model:str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    async def invoke(self,message:list[dict])->str:
        try:
            _llm = await self.llm.chat.completions.create(
                model=self.model,
                messages=message
            )
            return _llm.choices[0].message.content
        except Exception as e:
            return f"[LLM Error] {type(e).__name__}: {e}"



class ShortTermMemory:
    def __init__(self,max_rounds = 10):
        self.conversation_history = deque(maxlen=max_rounds)
        self.max_rounds = max_rounds

    def add_message(self,role:str,content:str):
        messages = {"role":role,"content":content}
        self.conversation_history.append(messages)

    def get_messages(self)->list[dict]:
        return list(self.conversation_history)



class Agent:
    def __init__(self,llm:LLM,system_prompt:str,max_rounds:int):
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = ShortTermMemory(max_rounds=max_rounds)

    async def chat(self,user_input:str)->str:
        self.memory.add_message("user",user_input)
        messages = [{"role":"system","content":self.system_prompt}]
        messages.extend(self.memory.get_messages())
        response = await self.llm.invoke(messages)
        self.memory.add_message("assistant",response)
        return response



async def main():
    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("qw_url")
    if not api_key or not base_url:
        raise ValueError("请设置 OPENAI_API_KEY 和 qw_url 环境变量")
    llm = LLM(api_key=api_key,base_url=base_url,model = "qwen3.7-flash-2026-07-15"
    )
    system_prompt = "你是一位高冷的ai助手，回复简短而精准，按照这个人设回复用户消息"
    agent = Agent(llm,system_prompt,max_rounds=10)
    loop = asyncio.get_running_loop()
    while True:
        user_input = await loop.run_in_executor(None, input, "\n请输入你的问题: ")
        result = await agent.chat(user_input)
        print(result)

if __name__ == "__main__":
    asyncio.run(main())