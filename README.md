# Diploma work: Smart POS & Loyalty

Дипломный проект (бакалавриат, AI in Business):
**«Разработка системы поддержки принятия решений на основе ИИ для оптимизации операционной деятельности сети кофеен»**.

Ядро проекта: **AI-ассистент бариста** с распознаванием лиц. Система узнаёт гостя на кассе, показывает
карточку и в реальном времени подсказывает бариста персональное предложение (upsell / cross-sell).
Пилотные точки: Espresso Day, Luna Coffee.

> Журнал работы и решения: [docs/PROGRESS.md](docs/PROGRESS.md) ·
> Концепт: [docs/CONCEPT.md](docs/CONCEPT.md) ·
> Структура диплома: [docs/THESIS_STRUCTURE.md](docs/THESIS_STRUCTURE.md)

## Архитектура

| Блок | Папка | Стек |
|---|---|---|
| Computer Vision: эмбеддинги, поиск лица | `backend/smartpos/faces.py` | OpenCV, InsightFace (ArcFace), FAISS |
| Recommendation Engine | `backend/smartpos/recommender.py` | pandas / scikit-learn (логистическая регрессия, гибрид) |
| Backend и бонусная логика | `backend/smartpos/main.py`, `db.py` | FastAPI, SQLite |
| Данные для прототипа | `backend/smartpos/synth.py` | синтетические чеки за 6 месяцев |
| Интерфейс бариста | `frontend/index.html` | HTML/JS |

## Экраны

- `http://localhost:8000/` (или `/kiosk`): **киоск для гостя** (планшет у стойки): узнаёт гостя, приветствует, предлагает напитки с учётом времени, сезона, праздников и акций, принимает оплату (демо).
- `http://localhost:8000/barista`: экран для персонала (карточка гостя и подсказки).

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m smartpos.synth        # создать синтетическую БД (при первом старте создаётся сама)
python -m smartpos.evaluate     # офлайн-метрики рекомендательной модели
pytest -q                       # тесты
uvicorn smartpos.main:app --reload   # http://localhost:8000
```

Реальное распознавание по камере: `SMARTPOS_FACE_ENGINE=insightface` (при первом запуске скачивается модель `buffalo_l`).
По умолчанию работает симуляция эмбеддингов, чтобы демо запускалось без камеры.

## Приватность

В базе хранятся только обезличенные эмбеддинги, не фотографии. Регистрация возможна только с согласием (opt-in),
согласие можно отозвать, и вектор лица удаляется (`DELETE /api/customers/{id}/biometrics`).
