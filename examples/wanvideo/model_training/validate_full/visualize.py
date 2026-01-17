import os
import json
import glob
import datetime
import shutil
import hashlib

# ================= 配置部分 =================

# 你的输出目录（结果所在的根目录）
ROOT_DIR = "/m2v_intern/mengzijie/DiffSynth-Studio/output/output_20260115_191302"

# 网页文件名
OUTPUT_HTML_NAME = "index.html"
# 图片转存的子文件夹名称
ASSETS_DIR_NAME = "assets_images"

# ================= CSS 样式 (保持美观) =================
STYLES = """
<style>
    :root {
        --bg-color: #121212;
        --card-bg: #1e1e1e;
        --text-main: #e0e0e0;
        --text-sub: #b0b0b0;
        --accent-pos: #4caf50;
        --accent-neg: #f44336;
        --border-color: #333;
    }
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--bg-color);
        color: var(--text-main);
        margin: 0;
        padding: 20px;
    }
    h1 { text-align: center; margin-bottom: 30px; color: #fff; font-weight: 300; }
    .stats { text-align: center; color: #888; margin-bottom: 20px; font-size: 0.9em; }
    .container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(580px, 1fr));
        gap: 25px;
        max-width: 1800px;
        margin: 0 auto;
    }
    .card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
    }
    .card-header {
        font-size: 1em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #90caf9;
        word-break: break-all;
        border-bottom: 1px solid #333;
        padding-bottom: 8px;
    }
    .media-row {
        display: flex;
        gap: 10px;
        height: 320px; /* 固定高度确保排版整齐 */
        margin-bottom: 15px;
    }
    .media-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #000;
        border: 1px solid #333;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    .media-label {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(0,0,0,0.6);
        color: #fff;
        font-size: 0.75em;
        padding: 4px;
        text-align: center;
        z-index: 2;
    }
    .media-col img, .media-col video {
        width: 100%;
        height: 100%;
        object-fit: contain; /* 保持比例 */
        display: block;
    }
    .prompt-section {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-size: 0.85em;
    }
    .prompt-box {
        padding: 8px;
        border-radius: 4px;
        max-height: 100px;
        overflow-y: auto;
        line-height: 1.3;
    }
    .pos { background: rgba(76, 175, 80, 0.1); border-left: 3px solid var(--accent-pos); }
    .neg { background: rgba(244, 67, 54, 0.1); border-left: 3px solid var(--accent-neg); }
    
    .label-text { font-weight: bold; opacity: 0.7; margin-right: 5px; }
    
    /* 滚动条美化 */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
"""

def get_unique_filename(path):
    """根据路径生成唯一的哈希文件名，避免不同文件夹下的同名文件冲突"""
    hash_object = hashlib.md5(path.encode())
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png" # 默认后缀
    return hash_object.hexdigest() + ext

def process_data():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ 错误: 根目录不存在 -> {ROOT_DIR}")
        return

    # 1. 准备图片存放目录
    assets_dir_path = os.path.join(ROOT_DIR, ASSETS_DIR_NAME)
    if not os.path.exists(assets_dir_path):
        os.makedirs(assets_dir_path)
        print(f"📂 创建资源文件夹: {assets_dir_path}")
    
    # 2. 读取所有 JSONL
    jsonl_files = glob.glob(os.path.join(ROOT_DIR, "*.jsonl"))
    data_list = []
    
    print(f"🔍 扫描到 {len(jsonl_files)} 个数据文件...")

    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        data_list.append(item)
                    except:
                        pass
    
    # 按时间倒序
    data_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # 3. 生成 HTML 内容
    html_cards = ""
    
    print(f"🚀 开始处理 {len(data_list)} 条记录，正在复制图片...")
    
    for idx, item in enumerate(data_list):
        video_name = item.get('video_name', '')
        # 视频在同级目录，直接引用
        video_src = video_name
        
        # 图片处理：复制到 assets 文件夹
        raw_image_path = item.get('image_path', '')
        local_image_name = "placeholder.png"
        
        if os.path.exists(raw_image_path):
            # 生成唯一文件名并复制
            unique_name = get_unique_filename(raw_image_path)
            target_path = os.path.join(assets_dir_path, unique_name)
            
            # 如果目标文件不存在，才复制（避免重复运行变慢）
            if not os.path.exists(target_path):
                try:
                    shutil.copy2(raw_image_path, target_path)
                except Exception as e:
                    print(f"   ⚠️ 复制图片失败: {e}")
            
            # HTML 中引用相对路径: assets_images/xxx.png
            local_image_src = f"{ASSETS_DIR_NAME}/{unique_name}"
        else:
            # 图片源文件不存在
            local_image_src = "" 
            print(f"   ⚠️ 原图不存在: {raw_image_path}")

        prompt = item.get('prompt', '')
        neg_prompt = item.get('negative_prompt', '')
        timestamp = item.get('timestamp', '')
        
        # 只有当图片路径有效时显示图片，否则显示错误提示
        img_tag = f'<img src="{local_image_src}" loading="lazy" onclick="window.open(this.src)">' if local_image_src else '<div style="padding:20px;text-align:center;color:#666;">原图丢失</div>'

        card = f"""
        <div class="card">
            <div class="card-header">{idx+1}. {video_name}</div>
            
            <div class="media-row">
                <div class="media-col">
                    <div class="media-label">Reference Image</div>
                    {img_tag}
                </div>
                <div class="media-col">
                    <div class="media-label">Generated Video</div>
                    <video controls preload="none" poster="{local_image_src}">
                        <source src="{video_src}" type="video/mp4">
                    </video>
                </div>
            </div>
            
            <div class="prompt-section">
                <div class="prompt-box pos">
                    <span class="label-text">PROMPT:</span>{prompt}
                </div>
                <div class="prompt-box neg">
                    <span class="label-text">NEGATIVE:</span>{neg_prompt}
                </div>
                <div style="text-align:right; color:#555; font-size:0.8em;">{timestamp}</div>
            </div>
        </div>
        """
        html_cards += card

    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>视频生成报告</title>
        {STYLES}
    </head>
    <body>
        <h1>生成结果可视化报告</h1>
        <div class="stats">
            生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | 
            数据量: {len(data_list)} | 
            目录: {os.path.basename(ROOT_DIR)}
        </div>
        <div class="container">
            {html_cards}
        </div>
    </body>
    </html>
    """
    
    output_html_path = os.path.join(ROOT_DIR, OUTPUT_HTML_NAME)
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print("="*50)
    print("✅ 处理完成！")
    print(f"📂 图片已缓存至: {assets_dir_path}")
    print(f"📄 网页已生成至: {output_html_path}")
    print("\n【使用方法】")
    print(f"cd {ROOT_DIR}")
    print("python -m http.server 8081")
    print("然后在浏览器打开: http://localhost:8081")
    print("="*50)

if __name__ == "__main__":
    process_data()