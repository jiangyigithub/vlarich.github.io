+++
date = '2026-04-08T16:38:44+08:00'
draft = true
title = 'Test'
+++
### step 1 新建文章
cd F:\vscode\hugo-root

hugo new posts/文章名字.md
### step 2 本地验证
hugo serve -D --disableFastRender
或
hugo serve

### step 3 更新 public 目录
hugo	#注意此时不能再 hugo serve了，否则public目录某些html的连接为http://localhost:1313/...

git add .

git commit -m "update"

git push origin master

 

cd public

git add .

git commit -m "update"

git push origin master
