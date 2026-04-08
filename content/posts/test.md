+++
date = '2026-04-08T16:38:44+08:00'
draft = true
title = 'Test'
+++
https://www.bilibili.com/video/BV1m4411c7ia?spm_id_from=333.788.videopod.episodes&vd_source=0d7a659e0c3fd86bc699b9150fa1cbbb&p=13
https://huuuuuuo.github.io/post/hugo%E6%97%A5%E5%B8%B8%E6%9B%B4%E6%96%B0%E6%B5%81%E7%A8%8B/
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
