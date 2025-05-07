# 终端运行方式：uv run clienttest.py http://0.0.0.0:8020/sse 
import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
import time
from mcp import ClientSession
from mcp.client.sse import sse_client
import PyPDF2
from openai import AsyncOpenAI
from dotenv import load_dotenv
from data_loader import LocalKnowledge
import base64

load_dotenv()

class MCPClient:
    def __init__(self, knowledge_path = "selectdata.xlsx"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
        # 显式初始化上下文属性
        self._session_context = None
        self._streams_context = None
        self.knowledge = LocalKnowledge(knowledge_path)
        self.pdf_text = ""
        self.current_image = None

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context is not None:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context is not None:
            await self._streams_context.__aexit__(None, None, None)

    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                if pdf_reader.is_encrypted:
                    print("\n**PDF文件已被加密，无法读取。")
                    return
                
                full_text = []
                total_pages = len(pdf_reader.pages)
                
                print(f"\n**正在处理PDF文件，共{total_pages}页...")
                
                for i, page in enumerate(pdf_reader.pages, 1):
                    print(f"正在提取第 {i}/{total_pages} 页... ", end='\r')
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)
                
                self.pdf_text = "\n".join(full_text)
                print(f"\n**PDF文本提取完成!")
                
        except FileNotFoundError:
            print(f"\n**错误：未找到文件 {pdf_path}")
        except Exception as e:
            print(f"\n**处理过程中发生错误：{str(e)}")

    async def analyze_image(self, image_path: str, question: str = "请分析这张图片的内容，并提供详细描述。") -> str:
        """Analyze image using OpenAI's Vision API"""
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            self.current_image = base64_image
            
            response = await self.openai.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"图片分析出错: {str(e)}"

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI API and available tools"""
        # 本地知识搜索
        local_results = self.knowledge.search(query)
        context = []

        if local_results:
            context.append("【本地知识库匹配结果】")
            for res in local_results:
                context.append(f"Q: {res['question']}\nA: {res['answer']}\n（相似度：{res['score']:.2f}）")

        if self.pdf_text:
            context.append("\n【PDF文档内容】")
            context.append(self.pdf_text)

        messages = [
            {
                "role": "system", 
                "content": "你是一个留学咨询专家，以下是相关问答参考：\n" + 
                          "\n".join(context) if context else "无本地知识匹配"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # 如果当前有加载的图片，将其添加到消息中
        if self.current_image and "图片" in query:
            messages[1]["content"] = [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.current_image}"
                    }
                }
            ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Initial OpenAI API call
        completion = await self.openai.chat.completions.create(
            model="gpt-4.1",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []
        
        assistant_message = completion.choices[0].message
        
        local_info = ""
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                if tool_name == "search_local_study_abroad_data":
                    tool_args = json.loads(tool_call.arguments)
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_results.append({"call": tool_name, "result": result})
                    local_info = result.content[0].text
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

        else: 
            if isinstance(assistant_message.content, (dict, list)):
                final_text.append(str(assistant_message.content))
            else:
                final_text.append(assistant_message.content)

        if local_results:  # 当存在本地匹配结果时
            if not tool_results:
                final_text.append("\n【科石小建议】")
                final_text.append(local_results[0]['answer'])

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries, 'pdf' to load a PDF file, 'image' to analyze an image, or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'pdf':
                    pdf_path = input("Enter PDF file path: ").strip()
                    self.extract_pdf_text(pdf_path)
                    print("PDF已加载，您现在可以询问PDF相关的问题")
                    continue
                elif query.lower() == 'image':
                    image_path = input("Enter image file path: ").strip()
                    print("请输入您对图片的问题(直接回车将执行默认分析):")
                    image_question = input().strip()
                    if not image_question:
                        image_question = "请分析这张图片的内容，并提供详细描述。"
                    analysis = await self.analyze_image(image_path, image_question)
                    print("\n图片分析结果:")
                    print(analysis)
                    print("图片已加载，您现在可以继续询问关于这张图片的问题")
                    continue
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())