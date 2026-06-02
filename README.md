# RL Mid-term Presentation

本项目是一个基于 [Slidev](https://sli.dev/) 的课程汇报幻灯片项目。Slidev 使用 Markdown 编写幻灯片内容，并支持代码块、公式、主题、布局、导出 PDF 等功能。

当前幻灯片入口文件是 `slides.md`，主题为 `@slidev/theme-seriph`。

## 环境准备

请先安装 Node.js，然后在项目目录中安装依赖：

```bash
npm install
```

项目依赖会安装 Slidev CLI、Seriph 主题以及 PDF 导出所需的 Chromium 支持。

## 本地预览

启动本地开发服务器：

```bash
npm run dev
```

命令启动后，终端会显示本地访问地址，通常是：

```text
http://localhost:3030
```

打开浏览器访问该地址即可预览幻灯片。修改 `slides.md` 后，页面会自动刷新。

## 如何修改幻灯片

主要修改文件：

```text
slides.md
```

Slidev 的每一页幻灯片用三个短横线分隔：

```markdown
---

# New Slide Title

Slide content here.
```

文件开头的 YAML frontmatter 用来配置主题、标题、转场、样式等：

```yaml
---
theme: seriph
title: Constraint-Conditioned Actor-Critic for Offline Safe RL
transition: slide-left
mdc: true
---
```

常见修改方式：

- 修改文字内容：直接编辑 Markdown 文本。
- 新增一页：在合适位置加入 `---`，然后写新页面内容。
- 修改页面布局：在页面开头添加布局配置，例如 `layout: center` 或 `layout: two-cols`。
- 编写公式：使用 LaTeX 语法，例如 `$$ ... $$`。
- 调整样式：可以直接使用 HTML 和 Tailwind CSS 类名，例如 `<div class="mt-8 text-xl">...</div>`。

## 导出 PDF

项目已经配置了 PDF 导出脚本：

```bash
npm run export
```

该命令等价于：

```bash
slidev export slides.md
```

导出完成后，默认会在项目目录生成：

```text
slides-export.pdf
```

如果导出失败，通常可以先确认依赖已经安装完整：

```bash
npm install
```

然后再次运行：

```bash
npm run export
```

## 导出 PPTX

可以使用 Slidev CLI 将当前幻灯片导出为 PowerPoint 文件：

```bash
npx slidev export slides.md --format pptx --output slides-export.pptx
```

导出完成后，会在项目目录生成：

```text
slides-export.pptx
```

注意：Slidev 导出的 PPTX 每页主要是图片形式，PowerPoint 中的文字通常不能直接编辑。

## 项目文件说明

```text
.
├── slides.md           # 幻灯片源文件
├── slides-export.pdf   # 已导出的 PDF
├── slides-export.pptx  # 已导出的 PPTX
├── package.json        # npm 脚本和依赖配置
├── package-lock.json   # npm 依赖锁定文件
└── pnpm-lock.yaml      # pnpm 依赖锁定文件
```

## 常用命令

```bash
# 安装依赖
npm install

# 本地预览
npm run dev

# 导出 PDF
npm run export

# 导出 PPTX
npx slidev export slides.md --format pptx --output slides-export.pptx
```
