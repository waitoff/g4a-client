#!/usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = 'szhdanoff@gmail.com'
# import base64
import os
import torch
import uuid
import random
import segno

from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageEnhance

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler,  # <-- Added import
    StableDiffusionPipeline
)
from diffusers.utils import load_image
from transformers import pipeline

# local import
# from load_lora import load_lora_weights
# from fix_image import enhance

from dotenv import load_dotenv

load_dotenv(override=True)
image_dir = os.environ.get("UPLOADER_IMAGE_DIR", "img")
image_height = int(os.environ.get("IMAGE_HEIGHT", 512))
image_width = int(os.environ.get("IMAGE_WIDTH", 512))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDN_URL = os.environ.get("CDN_URL", "https://871128.selcdn.ru/img/")

hf_home = os.getenv("HF_HOME")
if hf_home is None:
    hf_home = os.getenv("TRANSFORMERS_CACHE", os.path.join(os.getcwd(), "hf_home"))
    os.environ["HF_HOME"] = hf_home

os.makedirs(hf_home, exist_ok=True)


def get_device():
    """
        Get the device.

        Returns:
            The device object.
    """
    return device


async def text_to_image_simple(
        prompt: str,
        negative_prompt: str = "",
        model_id: str = "",
        image_file_name: str = "img",
        safety_checker: str = None,
        control_net: float = 0.8,  # controlnet
        initial_image_url: str = None,
):
    """
    Converts text prompt into an image using a local model.

    :param initial_image_url:
    :param control_net:
    :param prompt: The text prompt to convert into an image.
    :param negative_prompt: (Optional) The negative text prompt to generate a contrasting image.
    :param model_id: (Optional) The ID of the model to use for image generation.
    :param image_file_name: (Optional) The name of the generated image file.
    :param safety_checker: (Optional) The safety checker to use for the model.
    :return: A list of URLs pointing to the generated images.
    """

    os.makedirs(os.path.join(os.getcwd(), image_dir), exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16,
    ).to('cuda')

    pipe.enable_model_cpu_offload()

    if negative_prompt == '':
        negative_prompt = "cut off, bad, boring background, simple background, More_than_two_legs, " \
                          "more_than_two_arms, (3d render), (blender model), (((duplicate))), ((morbid)), " \
                          "((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), " \
                          "((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), " \
                          "(((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, " \
                          "extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), " \
                          "((missing legs)), ((extra arms)), ((extra legs)), mutated hands, (fused fingers), " \
                          "(too many fingers), ((long neck)), lowres, bad anatomy, bad hands, text, error, " \
                          "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, " \
                          "normal quality, jpeg artifacts, signature, watermark, username, blurry, artist's name"

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
    ).images

    nn = 0
    file_url = []
    for file in images:
        nn += 1
        file_name = f"{image_file_name}-{nn}.png"
        full_file_name = os.path.join(os.path.dirname(__file__), image_dir, file_name)
        file.save(full_file_name)
        file_url.append(f"{file_name}")

    return file_url


async def text_to_image_local(
        prompt: str,
        negative_prompt: str = "",
        model_id: str = "",
        image_file_name: str = "img",
        safety_checker: str = None,
        control_net: float = 0.8,  # controlnet
        initial_image_url: str = None,
):
    """
    Converts text prompt into an image using a local model.

    :param initial_image_url:
    :param control_net:
    :param prompt: The text prompt to convert into an image.
    :param negative_prompt: (Optional) The negative text prompt to generate a contrasting image.
    :param model_id: (Optional) The ID of the model to use for image generation.
    :param image_file_name: (Optional) The name of the generated image file.
    :param safety_checker: (Optional) The safety checker to use for the model.
    :return: A list of URLs pointing to the generated images.
    """
    def resize_for_condition_image(input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    if initial_image_url is not None:
        original_image = load_image(initial_image_url)
    else:
        original_image = load_image('init_img/zero.png')

    # image = np.array(original_image)
    # canny_image = resize_for_condition_image(original_image, 1024)
    canny_image = original_image
    # delete
    # canny_image.save("canny.png")

    os.makedirs(os.path.join(os.getcwd(), image_dir), exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        # "monster-labs/control_v1p_sdxl_qrcode_monster",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
                                        use_safetensors=True)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.enable_model_cpu_offload()

    if negative_prompt == '':
        negative_prompt = "cut off, bad, boring background, simple background, More_than_two_legs, " \
                          "more_than_two_arms, (3d render), (blender model), (((duplicate))), ((morbid)), " \
                          "((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), " \
                          "((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), " \
                          "(((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, " \
                          "extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), " \
                          "((missing legs)), ((extra arms)), ((extra legs)), mutated hands, (fused fingers), " \
                          "(too many fingers), ((long neck)), lowres, bad anatomy, bad hands, text, error, " \
                          "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, " \
                          "normal quality, jpeg artifacts, signature, watermark, username, blurry, artist's name"

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        # controlnet_conditioning_scale=0.5,
        # controlnet_conditioning_scale=0.8,
        controlnet_conditioning_scale=control_net,
        # guidance_scale=8.5,
    ).images

    nn = 0
    file_url = []
    for file in images:
        nn += 1
        file_name = f"{image_file_name}-{nn}.png"
        full_file_name = os.path.join(os.path.dirname(__file__), image_dir, file_name)
        file.save(full_file_name)
        file_url.append(f"{file_name}")

    return file_url


def text_to_image_image(
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float = 7.5,  # 0 - 50
        controlnet_conditioning_scale: float = 0.8,
        control_guidance_start: float = 0,  # 0 - 1
        control_guidance_end: float = 1,  #
        upscaler_strength: float = 1,  # 0 - 1
        seed: int = -1,
        sampler="DPM++ Karras SDE",
        # sampler="Euler",
        # progress=gr.Progress(track_tqdm=True),
        # profile: gr.OAuthProfile | None = None,
        image_file_name: str = "img",
        safety_checker: str = None,
        # need_images: int = 1,
):
    def transparent_image(image_file):
        image_file = image_file.convert("RGBA")
        datas = image_file.getdata()

        new_data = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)

        image_file.putdata(new_data)
        return image_file

    def prepare_image(image_file):
        new_image = image_file.resize((768, 768), resample=Image.LANCZOS)
        new_image = new_image.filter(ImageFilter.BLUR)
        new_image = new_image.filter(ImageFilter.UnsharpMask(radius=15, percent=500))
        new_image = new_image.filter(ImageFilter.SHARPEN)

        image_file = transparent_image(new_image)

        return image_file

    def center_crop_resize(img, output_size=(512, 512)):
        width, height = img.size

        # Calculate dimensions to crop to the center
        new_dimension = min(width, height)
        left = (width - new_dimension) / 2
        top = (height - new_dimension) / 2
        right = (width + new_dimension) / 2
        bottom = (height + new_dimension) / 2

        # Crop and resize
        img = img.crop((left, top, right, bottom))
        img = img.resize(output_size)
        return img

    def upscale(samples, upscale_method, scale_by):
        def common_upscale(samples, width, height, upscale_method, crop=False):
            if crop == "center":
                old_width = samples.shape[3]
                old_height = samples.shape[2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                x = 0
                y = 0
                if old_aspect > new_aspect:
                    x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
                elif old_aspect < new_aspect:
                    y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
                s = samples[:, :, y:old_height - y, x:old_width - x]
            else:
                s = samples

            return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        # s = samples.copy()
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
        return (s)

    def upload_image(image_file):
        """
        Uploads an image file to a remote SFTP server.

        :param image_file: The path to the image file to be uploaded.
        :type image_file: str
        """
        # host = os.getenv('SFTP_HOST', 'sftp.selcdn.ru')
        # port = os.getenv('SFTP_PORT', 22)
        # username = os.getenv('SFTP_USER', '252489_Tricia')
        # password = os.getenv('SFTP_PASS', '+(LiwM9yEf')
        # make_log = os.environ.get("UPLOADER_LOG", False)
        # print('-- Start file upload to sftp', image_file)
        #
        # opts = pysftp.CnOpts()
        # opts.hostkeys = None
        #
        # try:
        #     sftp = pysftp.Connection(host=host, port=port, username=username, password=password, log=make_log,
        #                              cnopts=opts)
        #     if make_log is True:
        #         print(f'-- {make_log}. Log file path: {sftp.logfile}')
        #
        #     if os.path.exists(image_file):
        #         with sftp.cd(image_dir):
        #             sftp.put(image_file)
        #         os.remove(image_file)
        #     sftp.close()
        #     print('-- Upload done...')
        #
        # except:
        #     print('-- Upload error...')
        #     sftp.close()

        return

    #

    os.makedirs(os.path.join(os.getcwd(), image_dir), exist_ok=True)
    base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
    # base_model = "Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE"
    # base_model = "digiplay/Landscape_PhotoReal_v1"
    # vae
    # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
    # vae = AutoencoderKL.from_single_file(url)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16,
        safety_checker=True,
        cache_dir=hf_home
    )
    # # controlnet = ControlNetModel.from_pretrained("DionTimmer/controlnet_qrcode-control_v1p_sd15",
    # #                                              torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster",
        torch_dtype=torch.float16,
        cache_dir=hf_home
    )  # , torch_dtype=torch.float16)

    main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        safety_checker=safety_checker,
        torch_dtype=torch.float16,
        cache_dir=hf_home
    ).to("cuda")

    SAMPLER_MAP = {
        "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True,
                                                                                   algorithm_type="sde-dpmsolver++"),
        "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    }

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)

    my_seed = random.randint(0, 2 ** 32 - 1) if seed == -1 else seed
    generator = torch.Generator(device="cuda").manual_seed(my_seed)

    # control_image = prepare_image(control_image)
    control_image_small = center_crop_resize(control_image)
    # control_image_small.save('control_image_small.png')
    control_image_large = center_crop_resize(control_image, (1024, 1024))
    # control_image_large.save('control_image_large.png')

    image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

    # image_pipe.safety_concept = (
    #     'an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, '
    #     'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, '
    #     'theft, vandalism, weapons, child abuse, brutality, cruelty, NSFW'
    # )

    # (You can use values between -5 and 5, but
    # I highly recommend using values around 0.4-0.6 to avoid changing the colors too much.)
    # lora_model_id = "models/lora/colorize.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)
    # # # #  add_detail - between 0.5 and 1 weight
    # lora_model_id = "models/lora/add_detail.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)

    # The recommended weight is about -2.0~2.0.
    # The higher the weight is, the brighter the images will be. (while the negative weight can decrease the brightness)
    # lora_model_id = "models/lora/add_brightness.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)
    #
    # # # Works better if u use good keywords like: dark studio, rim lighting, two tone lighting, dimly lit, low key etc.
    # lora_model_id = "epiNoiseoffset_v2.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)

    # # By adjusting the LoRA weight from -1.0 to 1.0, a gradual reduction of the clothing on the drawn objects can be achieved.
    # lora_model_id = "ClothingAdjuster3.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent",
    )

    upscaled_latents = upscale(out, "nearest-exact", 2)

    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=35,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        # height=768,
        # width=768,
        num_images_per_prompt=1,
    ).images

    nn = 0
    file_url = []
    rnd = str(uuid.uuid4())

    for file in out_image:
        nn += 1
        file_name = f"{rnd}-{image_file_name}-{nn}.png"
        full_file_name = os.path.join(os.path.dirname(__file__), image_dir, file_name)
        # use gfpgan
        # fixed_file = enhance(file)
        fixed_file = file
        fixed_file.save(full_file_name)
        # file.save(full_file_name)
        file_url.append(f"{CDN_URL}{file_name}")

        upload_image(full_file_name)

    # print(file_url)
    return file_url


# print(text_to_image_local(prompt="red cat", pipeline="red cat", model_id="darkstorm2150/Protogen_v2.2_Official_Release"))
# # res1 = (text_to_image_local(prompt="red cat", pipeline="red cat", model_id="darkstorm2150/Protogen_v2.2_Official_Release"))
# # res2 = (text_to_image_local(prompt="red cat", pipeline='', model_id='', image_file_name=slugify(text="red cat", allow_unicode=True)))
# print(text_to_image_local(prompt="red cat", pipeline='', model_id='', image_file_name='image0-1.png'))
def text_to_image_local_v2(
        prompt: str,
        negative_prompt: str,
        initial_image_url: str,
        image_file_name: str,
        # need_images: int = 1,
):
    if negative_prompt == '':
        negative_prompt = (
            "(((asian,NSFW,nudity))), cut off, bad, boring background, simple background,"
            "More_than_two_legs, more_than_two_arms,,((morbid)), (((duplicate)))"
            "((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), "
            "((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), "
            "(((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, "
            "extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), "
            "((missing legs)), ((extra arms)), ((extra legs)), mutated hands, (fused fingers), "
            "(too many fingers), ((long neck)), low resolution, bad anatomy, bad hands, text, error, "
            "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, watermark, username, artist's name"
        )

    image = load_image(initial_image_url)

    # pipe = pipeline("text-generation", model="Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator")
    # gen_prompt = pipe(prompt, max_length=70, num_return_sequences=1)
    # result = []
    # for gen in gen_prompt:
    #     prompt = gen.get('generated_text', 'cats in flowers')

    result = text_to_image_image(
        image_file_name=image_file_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=image,
        guidance_scale=7.5,
        upscaler_strength=1,
        controlnet_conditioning_scale=1.8,  # 0-5 , def = 1.8
        control_guidance_start=0,
        control_guidance_end=1,
        # need_images=need_images,
    )

    return result


async def text_to_qr(
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float = 7.5,  # 0 - 50
        controlnet_conditioning_scale: float = 0.8,
        control_guidance_start: float = 0,  # 0 - 1
        control_guidance_end: float = 1,  #
        upscaler_strength: float = 1,  # 0 - 1
        seed: int = -1,
        sampler="DPM++ Karras SDE",
        # sampler="Euler",
        # progress=gr.Progress(track_tqdm=True),
        # profile: gr.OAuthProfile | None = None,
        image_file_name: str = "img",
        safety_checker: str = None,
        # need_images: int = 1,
):
    def transparent_image(image_file):
        image_file = image_file.convert("RGBA")
        datas = image_file.getdata()

        new_data = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)

        image_file.putdata(new_data)
        return image_file

    def prepare_image(image_file):
        new_image = image_file.resize((768, 768), resample=Image.LANCZOS)
        new_image = new_image.filter(ImageFilter.BLUR)
        new_image = new_image.filter(ImageFilter.UnsharpMask(radius=15, percent=500))
        new_image = new_image.filter(ImageFilter.SHARPEN)

        image_file = transparent_image(new_image)

        return image_file

    def center_crop_resize(img, output_size=(512, 512)):
        width, height = img.size

        # Calculate dimensions to crop to the center
        new_dimension = min(width, height)
        left = (width - new_dimension) / 2
        top = (height - new_dimension) / 2
        right = (width + new_dimension) / 2
        bottom = (height + new_dimension) / 2

        # Crop and resize
        img = img.crop((left, top, right, bottom))
        img = img.resize(output_size)
        return img

    def upscale(samples, upscale_method, scale_by):
        def common_upscale(samples, width, height, upscale_method, crop=False):
            if crop == "center":
                old_width = samples.shape[3]
                old_height = samples.shape[2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                x = 0
                y = 0
                if old_aspect > new_aspect:
                    x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
                elif old_aspect < new_aspect:
                    y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
                s = samples[:, :, y:old_height - y, x:old_width - x]
            else:
                s = samples

            return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        # s = samples.copy()
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
        return (s)

    def upload_image(image_file):
        """
        Uploads an image file to a remote SFTP server.

        :param image_file: The path to the image file to be uploaded.
        :type image_file: str
        """
        # host = os.getenv('SFTP_HOST', 'sftp.selcdn.ru')
        # port = os.getenv('SFTP_PORT', 22)
        # username = os.getenv('SFTP_USER', '252489_Tricia')
        # password = os.getenv('SFTP_PASS', '+(LiwM9yEf')
        # make_log = os.environ.get("UPLOADER_LOG", False)
        # print('-- Start file upload to sftp', image_file)
        #
        # opts = pysftp.CnOpts()
        # opts.hostkeys = None
        #
        # try:
        #     sftp = pysftp.Connection(host=host, port=port, username=username, password=password, log=make_log,
        #                              cnopts=opts)
        #     if make_log is True:
        #         print(f'-- {make_log}. Log file path: {sftp.logfile}')
        #
        #     if os.path.exists(image_file):
        #         with sftp.cd(image_dir):
        #             sftp.put(image_file)
        #         os.remove(image_file)
        #     sftp.close()
        #     print('-- Upload done...')
        #
        # except:
        #     print('-- Upload error...')
        #     sftp.close()
        return

    # ##########################################

    os.makedirs(os.path.join(os.getcwd(), image_dir), exist_ok=True)
    base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
    # base_model = "Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE"
    # base_model = "digiplay/Landscape_PhotoReal_v1"
    # vae
    # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
    # vae = AutoencoderKL.from_single_file(url)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16,
        safety_checker=True,
        cache_dir=hf_home
    )
    # # controlnet = ControlNetModel.from_pretrained("DionTimmer/controlnet_qrcode-control_v1p_sd15",
    # #                                              torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster",
        torch_dtype=torch.float16,
        cache_dir=hf_home
    )  # , torch_dtype=torch.float16)

    main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        safety_checker=safety_checker,
        torch_dtype=torch.float16,
        cache_dir=hf_home
    ).to("cuda")

    SAMPLER_MAP = {
        "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True,
                                                                                   algorithm_type="sde-dpmsolver++"),
        "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    }

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)

    my_seed = random.randint(0, 2 ** 32 - 1) if seed == -1 else seed
    generator = torch.Generator(device="cuda").manual_seed(my_seed)

    # control_image = prepare_image(control_image)
    control_image_small = center_crop_resize(control_image)
    # control_image_small.save('control_image_small.png')
    control_image_large = center_crop_resize(control_image, (1024, 1024))
    # control_image_large.save('control_image_large.png')

    image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

    # image_pipe.safety_concept = (
    #     'an image showing hate, harassment, violence, suffering, humiliation, harm, suicide, '
    #     'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, '
    #     'theft, vandalism, weapons, child abuse, brutality, cruelty, NSFW'
    # )

    # (You can use values between -5 and 5, but
    # I highly recommend using values around 0.4-0.6 to avoid changing the colors too much.)

    # # ----------------------  deleted
    # lora_model_id = "models/lora/colorize.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)
    # # # # #  add_detail - between 0.5 and 1 weight
    # lora_model_id = "models/lora/add_detail.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)

    # The recommended weight is about -2.0~2.0.
    # The higher the weight is, the brighter the images will be. (while the negative weight can decrease the brightness)
    # lora_model_id = "models/lora/add_brightness.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)
    #
    # # # Works better if u use good keywords like: dark studio, rim lighting, two tone lighting, dimly lit, low key etc.
    # lora_model_id = "epiNoiseoffset_v2.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)

    # # By adjusting the LoRA weight from -1.0 to 1.0, a gradual reduction of the clothing on the drawn objects can be achieved.
    # lora_model_id = "ClothingAdjuster3.safetensors"
    # image_pipe = load_lora_weights(image_pipe, lora_model_id)

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent",
    )

    upscaled_latents = upscale(out, "nearest-exact", 2)

    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=35,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        # height=768,
        # width=768,
        num_images_per_prompt=1,
    ).images

    nn = 0
    file_url = []
    rnd = str(uuid.uuid4())

    for file in out_image:
        nn += 1
        # file_name = f"{rnd}-{image_file_name}-{nn}.png"
        file_name = f"{image_file_name}-{nn}.png"
        full_file_name = os.path.join(os.path.dirname(__file__), image_dir, file_name)
        # # use gfpgan
        # fixed_file = enhance(file)
        # fixed_file.save(full_file_name)
        file.save(full_file_name)
        # file_url.append(f"{CDN_URL}{file_name}")
        file_url.append(f"{file_name}")

        # upload_image(full_file_name)

    # print(file_url)
    return file_url


async def text_to_qr_local(
        prompt: str,
        negative_prompt: str,
        image_file_name: str,
        qr_code_text: str = None,
        controlnet: float = 1.8,
):
    # print(prompt, negative_prompt, image_file_name, qr_code_text)

    if negative_prompt == '':
        negative_prompt = "(((asian))),(((NSFW))),(((nudity)))"

        # negative_prompt = (
        #     "(((asian))),(((NSFW))),(((nudity))), cut off, bad, boring background, simple background,"
        #     "More_than_two_legs, more_than_two_arms,((morbid)),(((duplicate)))"
        #     "((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), "
        #     "((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), "
        #     "(((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, "
        #     "extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), "
        #     "((missing legs)), ((extra arms)), ((extra legs)), mutated hands, (fused fingers), "
        #     "(too many fingers), ((long neck)), low resolution, bad anatomy, bad hands, text, error, "
        #     "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, "
        #     "normal quality, jpeg artifacts, signature, watermark, username, artist's name"
        # )

    if qr_code_text is None:
        qr_code_text = 'https://smart-soft.pro'

    os.makedirs('img', exist_ok=True)
    qr = segno.make(qr_code_text, micro=False)
    # qr = segno.make(qr_code_text)
    qr.save('img/qrcode.png', scale=30)

    image = load_image('img/qrcode.png')

    result = await text_to_qr(
        image_file_name=image_file_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=image,
        # guidance_scale=7.5,
        guidance_scale=8.5,  # чем меньше число, тем больше "халявит" сетка
        upscaler_strength=1,
        controlnet_conditioning_scale=1.9,  # 0-5, def=1.8 # чем больше число - тем строже используется вторая картинка
        # controlnet_conditioning_scale=controlnet,
        control_guidance_start=0,
        control_guidance_end=1,
    )
    return result


def get_prompt(prompt):
    pipe = pipeline("text-generation", model="succinctly/text2image-prompt-generator")
    gen_prompt = pipe(prompt, max_length=77, num_return_sequences=1)
    new_prompt = gen_prompt[0].get('generated_text', '')
    return new_prompt
