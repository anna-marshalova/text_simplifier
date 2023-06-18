# Расширение для упрощения текста

Как воспрользоваться:
1. Клонируйте репозиторий: 
```commandline
git clone https://github.com/anna-marshalova/text_simplifier
```
2. Создайте файл `config.json` со следующим содержимым:
```json
{"api_url": "https://api-inference.huggingface.co/models/M-A-E/russian_text_simplification",
"huggingface_token": <YOUR TOKEN>}
```
На месте `<YOUR TOKEN>` нужно вставить ваш [токен из huggingface](https://huggingface.co/settings/tokens)
3. Зайдите на [страницу расширений в Chrome](chrome://extensions/)
4. Включите режим разработчика (в правом вернем углу)
5. Нажмите `Загрузить распакованное расширение` и выберите папку `extension` репозитория
6. Готово! Теперь можно пользоваться расширением:)

Настройки параметров генерации можно редактировать в файле `generation_config.json`.