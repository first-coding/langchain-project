import os  # 导入操作系统相关的模块
import shutil  # 导入高级文件操作模块

from app_modules.overwrites import postprocess  # 从app_modules.overwrites模块导入postprocess函数
from app_modules.presets import *  # 从app_modules.presets模块导入所有内容
from clc.langchain_application import LangChainApplication  # 从clc.langchain_application模块导入LangChainApplication类

class LangChainCFG:  # 定义LangChainCFG类
    llm_model_name = 'THUDM/chatglm-6b-int4-qe'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'GanymedeNil/text2vec-large-chinese'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = './cache'  # 向量存储路径
    docs_path = './docs'  # 文档存储路径
    kg_vector_stores = {  # 知识库向量存储路径字典
        '中文维基百科': './cache/zh_wikipedia',  # 中文维基百科向量存储路径
        '大规模金融研报': './cache/financial_research_reports',  # 大规模金融研报向量存储路径
        '初始化': './cache',  # 初始化向量存储路径
    } 
    patterns = ['模型问答', '知识库问答']  # 模式列表
    n_gpus=0  # 使用的GPU数量

def upload_file(file):  # 定义上传文件函数
    if not os.path.exists("docs"):  # 如果docs目录不存在
        os.mkdir("docs")  # 创建docs目录
    filename = os.path.basename(file.name)  # 获取文件名
    shutil.move(file.name, "docs/" + filename)  # 移动文件到docs目录
    file_list.insert(0, filename)  # 将文件名插入文件列表的开头
    application.source_service.add_document("docs/" + filename)  # 向应用程序的source_service添加文档
    return gr.Dropdown(choices=file_list, value=filename, interactive=True)  # 返回一个带有文件列表的下拉菜单

def get_file_list():  # 定义获取文件列表函数
    if not os.path.exists("docs"):  # 如果docs目录不存在
        return []  # 返回空列表
    return [f for f in os.listdir("docs")]  # 返回docs目录中的文件列表

def set_knowledge(kg_name, history):  # 定义设置知识库函数
    try:
        application.source_service.load_vector_store(config.kg_vector_stores[kg_name])  # 加载指定知识库的向量存储
        msg_status = f'{kg_name}知识库已成功加载'  # 设置成功加载的消息状态
    except Exception as e:  # 捕获异常
        print(e)  # 打印异常信息
        msg_status = f'{kg_name}知识库未成功加载'  # 设置未成功加载的消息状态
    return history + [[None, msg_status]]  # 返回更新后的历史记录

def clear_session():  # 定义清除会话函数
    return '', None  # 返回空字符串和None

def predict(input,  # 定义预测函数
            large_language_model,  # 大型语言模型
            embedding_model,  # 嵌入模型
            top_k,  # 检索top-k文档
            use_web,  # 是否使用网络搜索
            use_pattern,  # 使用的模式
            history=None):  # 历史记录，默认为None
    print(large_language_model, embedding_model)  # 打印大型语言模型和嵌入模型
    print(input)  # 打印输入
    if history == None:  # 如果历史记录为None
        history = []  # 初始化为空列表

    if use_web == '使用':  # 如果使用网络搜索
        web_content = application.source_service.search_web(query=input)  # 搜索网络内容
    else:
        web_content = ''  # 否则网络内容为空
    search_text = ''  # 初始化搜索文本为空
    if use_pattern == '模型问答':  # 如果使用模型问答模式
        result = application.get_llm_answer(query=input, web_content=web_content)  # 获取大型语言模型的答案
        history.append((input, result))  # 将输入和结果添加到历史记录
        search_text += web_content  # 添加网络内容到搜索文本
        return '', history, history, search_text  # 返回空字符串、历史记录、历史记录和搜索文本

    else:  # 否则使用知识库问答模式
        resp = application.get_knowledge_based_answer(  # 获取基于知识库的答案
            query=input,  # 查询输入
            history_len=1,  # 历史长度为1
            temperature=0.1,  # 温度参数
            top_p=0.9,  # top-p参数
            top_k=top_k,  # top-k参数
            web_content=web_content,  # 网络内容
            chat_history=history  # 聊天历史记录
        )
        history.append((input, resp['result']))  # 将输入和结果添加到历史记录
        for idx, source in enumerate(resp['source_documents'][:4]):  # 遍历前4个源文档
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'  # 设置分隔符
            search_text += f'{sep}\n{source.page_content}\n\n'  # 添加源文档内容到搜索文本
        print(search_text)  # 打印搜索文本
        search_text += "----------【网络检索内容】-----------\n"  # 添加网络检索内容分隔符
        search_text += web_content  # 添加网络内容到搜索文本
        return '', history, history, search_text  # 返回空字符串、历史记录、历史记录和搜索文本
    
if __name__ == '__main__':  # 主程序
    config = LangChainCFG()  # 创建LangChainCFG实例
    application = LangChainApplication(config)  # 创建LangChainApplication实例
    application.source_service.init_source_vector()  # 初始化source_service的向量

    file_list = get_file_list()  # 获取文件列表

    with open("assets/custom.css", "r", encoding="utf-8") as f:  # 打开自定义CSS文件
        customCSS = f.read()  # 读取CSS内容

    with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:  # 创建Gradio Blocks实例
        gr.Markdown("""<h1><center>LangChain</center></h1>
            <center><font size=3>
            </center></font>
            """)  # 添加Markdown内容
        state = gr.State()  # 创建Gradio State实例

        with gr.Row():  # 创建Gradio Row实例
            with gr.Column(scale=1):  # 创建Gradio Column实例，比例为1
                embedding_model = gr.Dropdown([  # 创建嵌入模型下拉菜单
                    "text2vec-base"
                ],
                    label="Embedding model",  # 标签为"Embedding model"
                    value="text2vec-base")  # 默认值为"text2vec-base"

                large_language_model = gr.Dropdown(["ChatGLM-6B-int4",],label="large language model",value="ChatGLM-6B-int4")  # 创建大型语言模型下拉菜单

                top_k = gr.Slider(1,20,value=4,step=1,label="检索top-k文档",interactive=True)  # 创建top-k滑块

                use_web = gr.Radio(["使用", "不使用"], label="web search",info="是否使用网络搜索，使用时确保网络通常",value="不使用")  # 创建网络搜索单选按钮
                use_pattern = gr.Radio(['模型问答','知识库问答',],label="模式",value='模型问答',interactive=True)  # 创建模式单选按钮

                kg_name = gr.Radio(list(config.kg_vector_stores.keys()),label="知识库",value=None,info="使用知识库问答，请加载知识库",interactive=True)  # 创建知识库单选按钮
                set_kg_btn = gr.Button("加载知识库")  # 创建加载知识库按钮

                file = gr.File(label="将文件上传到知识库库，内容要尽量匹配",visible=True,file_types=['.txt', '.md', '.docx', '.pdf'])  # 创建文件上传控件

            with gr.Column(scale=4):  # 创建Gradio Column实例，比例为4
                with gr.Row():  # 创建Gradio Row实例
                    chatbot = gr.Chatbot(label='Langchain',height=400)  # 创建聊天机器人控件
                with gr.Row():  # 创建Gradio Row实例
                    message = gr.Textbox(label='请输入问题')  # 创建文本框控件
                with gr.Row():  # 创建Gradio Row实例
                    clear_history = gr.Button("🧹 清除历史对话")  # 创建清除历史对话按钮
                    send = gr.Button("🚀 发送")  # 创建发送按钮
                with gr.Row():  # 创建Gradio Row实例
                    gr.Markdown("""提醒：<br>
                                            [LangChain](https://github.com/first-coding/langchain-project) <br>
                                            有任何使用问题[Github Issue区](https://github.com/first-coding/langchain-project)进行反馈. <br>
                                            """)  # 添加Markdown内容
            with gr.Column(scale=2):  # 创建Gradio Column实例，比例为2
                search = gr.Textbox(label='搜索结果')  # 创建搜索结果文本框

            file.upload(upload_file,inputs=file,outputs=None)  # 设置文件上传事件

            set_kg_btn.click(set_knowledge,show_progress=True,inputs=[kg_name, chatbot],outputs=chatbot)  # 设置加载知识库按钮点击事件

            send.click(predict,inputs=[message,large_language_model,embedding_model,top_k,use_web,use_pattern,state],outputs=[message, chatbot, state, search])  # 设置发送按钮点击事件

            clear_history.click(fn=clear_session,inputs=[],outputs=[chatbot, state],queue=False)  # 设置清除历史对话按钮点击事件

            message.submit(predict,inputs=[message,large_language_model,embedding_model,top_k,use_web,use_pattern,state],outputs=[message, chatbot, state, search])  # 设置消息提交事件

    demo.launch(  # 启动Gradio应用
        server_name='0.0.0.0',  # 服务器名称
        server_port=8888,  # 服务器端口
        share=True,  # 共享
        show_error=True,  # 显示错误
        debug=True,  # 调试模式
        inbrowser=True,  # 在浏览器中打开
    )
