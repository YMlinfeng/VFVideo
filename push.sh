#!/bin/bash

# 安全版本：先备份再操作

# 提示用户
echo "警告：此操作将永久删除所有git历史记录！"
echo "当前目录: $(pwd)"
echo ""
read -p "是否继续？(y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 1
fi

# 备份当前目录（可选）
echo "正在备份..."
backup_dir="../$(basename $(pwd))_backup_$(date +%Y%m%d_%H%M%S)"
cp -r . "$backup_dir"
echo "已备份到: $backup_dir"

# 保存当前文件列表
echo "当前文件列表:"
find . -name ".git" -prune -o -type f -print | head -20

# 移除.git文件夹
echo "正在移除.git文件夹..."
rm -rf .git

# 初始化新仓库
echo "正在初始化新的git仓库..."
git init

# 配置用户信息
git config user.name "YMlinfeng"
git config user.email "xiao102851@163.com"

# 添加文件并提交
echo "正在添加文件..."
git add .

echo "正在创建初始提交..."
git commit -m "Initial commit from YMlinfeng"

# 添加远程仓库
echo "正在添加远程仓库..."
git remote add origin https://github.com/YMlinfeng/VFVideo.git

# 尝试推送到远程
echo "正在推送到远程仓库..."
if git branch --show-current | grep -q "master"; then
    git push -u origin master --force
elif git branch --show-current | grep -q "main"; then
    git push -u origin main --force
else
    # 如果没有分支，创建main分支
    git branch -M main
    git push -u origin main --force
fi

echo ""
echo "操作完成！"
echo "新仓库地址: https://github.com/YMlinfeng/VFVideo"
echo "旧的历史记录已被完全清除。"