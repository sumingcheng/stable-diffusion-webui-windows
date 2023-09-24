# 导入必要的库和模块
from contextlib import closing  # 用于资源管理，确保资源如文件被正确关闭
import modules.scripts  # 导入scripts模块
from modules import processing  # 导入processing模块
from modules.generation_parameters_copypaste import create_override_settings_dict  # 导入特定的函数
from modules.shared import opts, cmd_opts  # 导入共享的配置选项
import modules.shared as shared  # 导入shared模块并为其设置别名
from modules.ui import plaintext_to_html  # 导入文本到HTML的转换函数
import gradio as gr  # 导入gradio库，用于创建Web UI


# 定义txt2img函数，该函数将文本转换为图像
def txt2img(
        id_task: str,
        prompt: str,
        negative_prompt: str,
        prompt_styles,
        steps: int,
        sampler_name: str,
        n_iter: int,
        batch_size: int,
        cfg_scale: float,
        height: int,
        width: int,
        enable_hr: bool,
        denoising_strength: float,
        hr_scale: float,
        hr_upscaler: str,
        hr_second_pass_steps: int,
        hr_resize_x: int,
        hr_resize_y: int,
        hr_checkpoint_name: str,
        hr_sampler_name: str,
        hr_prompt: str,
        hr_negative_prompt,
        override_settings_texts,
        request: gr.Request,
        *args
):
    # 根据提供的文本创建设置字典
    override_settings = create_override_settings_dict(override_settings_texts)

    # 创建一个处理对象，用于处理文本到图像的转换
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    # 设置脚本和参数
    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    # 设置用户信息
    p.user = request.username

    # 如果启用了控制台提示，则打印提示信息
    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    # 使用资源管理器确保处理对象在使用后被正确关闭
    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    # 清除进度条
    shared.total_tqdm.clear()

    # 获取生成信息
    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    # 如果设置了不显示图像，则清空图像列表
    if opts.do_not_show_images:
        processed.images = []

    # 返回处理后的图像、生成信息和其他相关信息
    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(
        processed.comments, classname="comments")
