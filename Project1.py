import os
from openai import OpenAI
#for acessing API from OpenAI

# 初始化OpenAI客户端
client = OpenAI(#create a concrete OpenAi client, allowing us to 
                #interact with OpenAI 
    api_key=os.getenv("DASHSCOPE_API_KEY"),#using os.getenv, getting API key from environment
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 用于存储多轮对话的历史
conversation_history = []

# 判断用户是否表达了结束对话的意图
def check_end_conversation(user_input):
    end_phrases = ["再见", "谢谢", "结束对话", "不需要了"]
    return any(phrase in user_input for phrase in end_phrases)

# 主对话逻辑
def chat_with_ai(user_input):
    global conversation_history
    
    # 将用户输入加入对话历史
    conversation_history.append({"role": "user", "content": user_input})
    
    # 检查用户是否希望结束对话
    if check_end_conversation(user_input):
        print("AI: 感谢您的咨询，再见")
        conversation_history = []  # 重置对话历史
        return
    
    # 生成AI回复
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=conversation_history
    )
    
    # 获取并打印AI的回复
    ai_response = completion.choices[0].message.content
    print("AI:", ai_response)
    
    # 将AI回复加入对话历史
    conversation_history.append({"role": "assistant", "content": ai_response})

# 主程序循环
while True:
    user_input = input("你: ")
    if user_input.lower() == "退出":
        print("对话已退出")
        break
    chat_with_ai(user_input)