## COMMANDLINE_ARGS

1. **--api**:
   - 这个参数允许你在API模式下运行`stable-diffusion-webui`。在此模式下，可以通过HTTP请求来与应用交互，而不是通过其图形用户界面。
2. **--xformers**:
   - 这个参数似乎是用来启用或访问某种特定功能或库，但文档中并未明确说明其确切用途。
3. **--opt-sdp-no-mem-attention**:
   - 此参数的确切功能未在文档中明确说明，可能是用于优化或配置内存注意机制的参数。
4. **--enable-insecure-extension-access**:
   - 这个参数允许不安全的扩展访问，可能是用于开发或测试目的。
5. **--nowebui**:
   - 此参数禁用Web用户界面，可能使`stable-diffusion-webui`仅作为后端服务运行，而不提供用户界面。
6. **COMMANDLINE_ARGS**:
   - 这个环境变量允许你传递附加的命令行参数给主程序[1](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Troubleshooting#:~:text=Troubleshooting,more gpu vram than ram)。
7. **--exit**:
   - 此参数使程序在安装后终止[2](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)。
8. **--data-dir DATA_DIR**:
   - 此参数设置存储所有用户数据的基本路径[2](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)。
9. **--config CONFIG**:
   - 通过此参数，可以指定构建模型的配置文件的路径[2](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)。
10. **--ckpt CKPT**:
    - 此参数允许你指定Stable Diffusion模型的检查点，如果指定，此检查点将被添加到检查点列表中并被加载[2](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)。