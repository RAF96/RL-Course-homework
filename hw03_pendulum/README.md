# RL Homework

### How to submit
Задачи отправляются боту в Telegram: `@RL_hw_bot`. В качестве решения задачи принимается zip-архив с кодом. Важно: содержимое папки с задание должно быть в корне, иначе бот не найдет ваше решение. Например, для первого задания, находящегося в папке `hw01_mountain_car`, все файлы из этой папки должны находиться в корне архива.

### Task
В данном задании необходимо обучить агента побеждать в игре Pendulum при помощи метода REINFORCE или A2C. Для решения задачи можно трансформировать состояние и награду среды.
К заданию также нужно приложить код обучения агента (не забудьте зафиксировать seed!), готовый (уже обученный) агент должен быть описан в классе Agent в файле `agent.py`.


### Оценка:
От 1 до 10, баллы начисляются за полученные агентом очки в среднем за 75 эпизодов. Максимальный балл соответствует `-150` очкам и выше, минимальный - `-400`