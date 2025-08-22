import os
import json
import random
import requests
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройки
COMFYUI_API_URL = "http://localhost:8188"
BOT_TOKEN = "" # Замените на свой токен
WORKFLOW_FILE = "flux_schnell.json" 

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка и сохранение workflow
def save_workflow(workflow):
    with open(WORKFLOW_FILE, 'w') as f:
        json.dump(workflow, f, indent=2)

def load_workflow():
    if os.path.exists(WORKFLOW_FILE):
        with open(WORKFLOW_FILE, 'r') as f:
            return json.load(f)
    return None

# Основной workflow
DEFAULT_WORKFLOW = {
    "6": {"inputs": {"text": "Replace this text with promt", "clip": ["38", 0]}, "class_type": "CLIPTextEncode"},
    "8": {"inputs": {"samples": ["31", 0], "vae": ["40", 0]}, "class_type": "VAEDecode"},
    "9": {"inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}, "class_type": "SaveImage"},
    "27": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
    "31": {"inputs": {"seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, 
                     "model": ["39", 0], "positive": ["6", 0], "negative": ["33", 0], "latent_image": ["27", 0]}, 
          "class_type": "KSampler"},
    "33": {"inputs": {"text": "", "clip": ["38", 0]}, "class_type": "CLIPTextEncode"},
    "38": {"inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", 
                     "type": "flux", "device": "default"}, "class_type": "DualCLIPLoader"},
    "39": {"inputs": {"unet_name": "flux1-schnell-fp8-e4m3fn.safetensors", "weight_dtype": "default"}, 
          "class_type": "UNETLoader"},
    "40": {"inputs": {"vae_name": "flux_vae.safetensors"}, "class_type": "VAELoader"}
}

# Проверяем и сохраняем workflow
if not load_workflow():
    save_workflow(DEFAULT_WORKFLOW)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я бот использующий flux для генерации изображений, отправь промт на английском для генерации."
    )

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    prompt_text = update.message.text
    logger.info(f"Запрос от {user.first_name}: {prompt_text}")

    status_msg = await update.message.reply_text("🔄 Генерация началась...")

    try:
        workflow = load_workflow()
        
        # Обновляем параметры
        workflow["6"]["inputs"]["text"] = prompt_text
        workflow["31"]["inputs"]["seed"] = random.randint(0, 10**18)
        
        # Отправка в ComfyUI API с увеличенным таймаутом
        response = requests.post(
            f"{COMFYUI_API_URL}/prompt", 
            json={"prompt": workflow},
            timeout=120  # Увеличиваем таймаут для длительных операций
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        
        await status_msg.edit_text("⏳ Обработка изображения...")
        while True:
            # Проверяем статус каждые 3 секунды
            history_response = requests.get(
                f"{COMFYUI_API_URL}/history/{prompt_id}",
                timeout=30
            )
            history = history_response.json()
            
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed"):
                    break
                if status.get("error"):
                    error_msg = status["error"].get("error_message", "Unknown error")
                    raise Exception(f"ComfyUI error: {error_msg}")
            
            await asyncio.sleep(3)
        
        # Получаем информацию о сгенерированном изображении
        output = history[prompt_id]["outputs"].get("9", {})
        if output and output.get("images"):
            image_info = output["images"][0]
            
            # Формируем параметры для скачивания
            params = {
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": "output"
            }
            
            # Скачиваем изображение с сервера ComfyUI
            image_response = requests.get(
                f"{COMFYUI_API_URL}/view",
                params=params,
                timeout=30
            )
            image_response.raise_for_status()
            
            # Отправляем изображение как бинарные данные
            await update.message.reply_photo(
                photo=image_response.content,
                caption=f"✅ Результат для: {prompt_text}"
            )
            await status_msg.delete()
        else:
            await status_msg.edit_text("❌ Ошибка: изображение не сгенерировано")
            
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        await status_msg.edit_text(f"❌ Ошибка генерации: {str(e)}")

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))
    
    logger.info("Бот запущен...")
    application.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
