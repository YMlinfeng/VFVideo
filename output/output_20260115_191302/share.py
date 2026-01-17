import gradio as gr
import os

# 设置为你的 output 目录路径
ROOT_DIR = "/m2v_intern/mengzijie/DiffSynth-Studio/output/output_20260115_191302"

def serve_file(path):
    # 简单的文件服务逻辑
    if path == "" or path == "/":
        path = "index.html"
    
    full_path = os.path.join(ROOT_DIR, path.lstrip("/"))
    if os.path.exists(full_path):
        return full_path
    return None

# 创建一个极简的 Gradio 应用来代理静态文件
# 注意：这只是为了利用 Gradio 的 share=True 功能
with gr.Blocks() as demo:
    gr.HTML(f"""
    <iframe src="file=index.html" width="100%" height="1000px" style="border:none;"></iframe>
    """)

# 启动并开启分享
# allowed_paths 允许 Gradio 访问该目录下的静态文件（图片/视频）
demo.launch(share=True, allowed_paths=[ROOT_DIR], server_port=8082)