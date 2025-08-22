import os
import json
import random
import requests
import logging
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройки
COMFYUI_API_URL = "http://localhost:8188"
BOT_TOKEN = ""# Замените на свой токен
WORKFLOW_FILE = "wan_video_workflow.json"

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Workflow для генерации видео
VIDEO_WORKFLOW = {
    "3": {
        "inputs": {
            "seed": 82628696717253,
            "steps": 30,
            "cfg": 6,
            "sampler_name": "uni_pc",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["48", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["40", 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"}
    },
    "6": {
        "inputs": {
            "text": "REPLACE THIS TEXT WITH PROMPT",
            "clip": ["38", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
    },
    "7": {
        "inputs": {
            "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "clip": ["38", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
    },
    "8": {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["39", 0]
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "Декодировать VAE"}
    },
    "28": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "fps": 16,
            "lossless": False,
            "quality": 90,
            "method": "default",
            "images": ["8", 0]
        },
        "class_type": "SaveAnimatedWEBP",
        "_meta": {"title": "Сохранить анимированный WEBP"}
    },
    "37": {
        "inputs": {
            "unet_name": "wan2.1_t2v_1.3B_fp16.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Загрузить модель диффузии"}
    },
    "38": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default"
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Загрузить CLIP"}
    },
    "39": {
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {"title": "Загрузить VAE"}
    },
    "40": {
        "inputs": {
            "width": 832,
            "height": 480,
            "length": 33,
            "batch_size": 1
        },
        "class_type": "EmptyHunyuanLatentVideo",
        "_meta": {"title": "Пустой HunyuanLatentVideo"}
    },
    "48": {
        "inputs": {
            "shift": 8,
            "model": ["37", 0]
        },
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "Выборка модели SD3"}
    }
}

def save_workflow(workflow):
    with open(WORKFLOW_FILE, 'w') as f:
        json.dump(workflow, f, indent=2)

def load_workflow():
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, 'r') as f:
            return json.load(f)
    return None

if not load_workflow():
    save_workflow(VIDEO_WORKFLOW)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я бот для генерации видео с помощью WAN модели. Отправь промт на английском для генерации видео."
    )

async def generate_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    prompt_text = update.message.text
    logger.info(f"Запрос от {user.first_name}: {prompt_text}")

    status_msg = await update.message.reply_text("🔄 Генерация видео началась...")

    try:
        workflow = load_workflow() or VIDEO_WORKFLOW
        
        # Обновляем параметры
        workflow["6"]["inputs"]["text"] = prompt_text
        workflow["3"]["inputs"]["seed"] = random.randint(0, 10**18)
        
        # Отправка в ComfyUI API
        response = requests.post(
            f"{COMFYUI_API_URL}/prompt", 
            json={"prompt": workflow},
            timeout=300
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        
        await status_msg.edit_text("⏳ Обработка видео...")
        while True:
            history_response = requests.get(
                f"{COMFYUI_API_URL}/history/{prompt_id}",
                timeout=60
            )
            history = history_response.json()
            
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed"):
                    break
                if status.get("error"):
                    error_msg = status["error"].get("error_message", "Unknown error")
                    raise Exception(f"ComfyUI error: {error_msg}")
            
            await asyncio.sleep(5)
        
        # Получаем информацию о сгенерированном видео
        output = history[prompt_id]["outputs"].get("28", {})
        if output and output.get("images"):
            image_info = output["images"][0]
            
            params = {
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": "output"
            }
            
            # Скачиваем анимированный WEBP
            video_response = requests.get(
                f"{COMFYUI_API_URL}/view",
                params=params,
                timeout=60
            )
            video_response.raise_for_status()
            
            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as temp_file:
                temp_file.write(video_response.content)
                temp_path = temp_file.name

            # Отправляем анимированный WEBP как документ
            with open(temp_path, 'rb') as video_file:
                await update.message.reply_document(
                    document=video_file,
                    caption=f"✅ Видео для: {prompt_text}",
                    filename="animation.webp"
                )
            
            # Удаляем временный файл
            os.unlink(temp_path)
            await status_msg.delete()
            
        else:
            await status_msg.edit_text("❌ Ошибка: видео не сгенерировано")
            
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        await status_msg.edit_text(f"❌ Ошибка генерации: {str(e)}")

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_video))
    
    logger.info("Бот запущен...")
    application.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
