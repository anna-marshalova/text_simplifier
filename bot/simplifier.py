import json
import requests
import logging
from pathlib import Path
import nltk

nltk.download('punkt')

logging.basicConfig(level=logging.DEBUG)


class Simplifier:
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.json"
        with open(self.config_path, "r") as file:
            self.config = json.load(file)
        self.default_generation_config_path = Path(__file__).parent / "generation_config.json"
        with open(self.default_generation_config_path, "r") as file:
            self.default_generation_config = json.load(file)
        self.HUGGINGFACE_TOKEN = self.config['huggingface_token']
        self.API_URL = self.config['api_url']
        self.headers = {"Authorization": f"Bearer {self.HUGGINGFACE_TOKEN}"}
        self.max_length = lambda x: len(x.split()) * self.default_generation_config['max_length_factor']

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

    def simplify_sent(self, sent):
        """
        Упрощение одного предложения
        :param sent: Предложение
        :return: Упрощеннное предложение
        """
        output = self.query({
            "inputs": sent,
            "parameters": {"do_sample": self.default_generation_config['do_sample'],
                           "repetition_penalty": self.default_generation_config['repetition_penalty'],
                           "max_length" : self.max_length(sent)
                           }
        })
        if 'error' in output:
            logging.debug(output)
            return sent
        return output[0]['generated_text']

    def simplify(self, text):
        """
        Упрощение текста
        :param text: Текст
        :return: Упрощеннный текст
        """
        # делим текст на предложения и упрощаем отдельно, т.к. модель была обучена на упрощении предложений
        sents = nltk.sent_tokenize(text)
        simplified_sents = [self.simplify_sent(sent) for sent in sents]
        simplified_text = ' '.join(simplified_sents)
        if simplified_text:
            return simplified_text
        return text


if __name__ == '__main__':
    simplifier = Simplifier()
    text = '14 декабря 1944 года рабочий посёлок Ички был переименован в рабочий посёлок Советский, после чего поселковый совет стал называться Советским.'
    text = 'Rocksteady Studios задумала создать Arkham City ещё во время разработки Arkham Asylum, оставив намёки на сиквел в игре и начав полноценное создание в феврале 2009 года. «Аркхэм-Сити» в пять раз превышает площадь лечебницы Аркхэм, а его дизайн был изменён, чтобы Бэтмен мог во время планирования пикировать. На маркетинговую кампанию было потрачено более года и десяти миллионов долларов, а выпуск игры сопровождался двумя музыкальными альбомами — один содержал партитуру от Рона Фиша и Ника Арундела, а другой — одиннадцать оригинальных песен от различных популярных исполнителей. Большинство персонажей озвучены актёрами, принимавшие участие в Arkham Asylum и анимационных проектах от DC.'
    print(simplifier.simplify(text))
