# Инструкция по развертыванию API сервера на Ubuntu

Полное руководство по установке и настройке CSV Analysis Agent API на сервере Ubuntu.

---

## Требования

- Ubuntu 20.04 или новее
- Python 3.8+
- Минимум 2GB RAM
- 10GB свободного места на диске
- Доступ к интернету
- Sudo права

---

## Часть 1: Подготовка сервера

### 1.1 Обновление системы

```bash
# Подключитесь к серверу по SSH
ssh your_user@your_server_ip

# Обновите пакеты
sudo apt update
sudo apt upgrade -y
```

### 1.2 Установка Python и необходимых пакетов

```bash
# Установка Python 3 и pip
sudo apt install -y python3 python3-pip python3-venv

# Установка дополнительных пакетов
sudo apt install -y git nginx supervisor

# Проверка версии Python
python3 --version  # Должно быть 3.8+
```

### 1.3 Создание пользователя для приложения (рекомендуется)

```bash
# Создать пользователя csvagent
sudo useradd -m -s /bin/bash csvagent

# Переключиться на этого пользователя
sudo su - csvagent
```

---

## Часть 2: Установка приложения

### 2.1 Клонирование репозитория

```bash
# Перейти в домашнюю директорию
cd ~

# Клонировать репозиторий (замените на ваш URL)
git clone https://github.com/your-username/Claude_code.git

# Перейти в директорию
cd Claude_code
```

### 2.2 Создание виртуального окружения

```bash
# Создать виртуальное окружение
python3 -m venv venv

# Активировать
source venv/bin/activate

# Обновить pip
pip install --upgrade pip
```

### 2.3 Установка зависимостей

```bash
# Установить зависимости для API
pip install -r requirements_api.txt

# Проверить установку
pip list
```

### 2.4 Настройка переменных окружения

```bash
# Создать файл .env
nano .env
```

Добавьте в файл:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
HOST=0.0.0.0
PORT=8000
```

Сохраните (Ctrl+O, Enter, Ctrl+X)

```bash
# Установите правильные права
chmod 600 .env
```

### 2.5 Тестовый запуск

```bash
# Запустите сервер для проверки
python api_server.py
```

Откройте в браузере: `http://your_server_ip:8000`

Должны увидеть JSON ответ: `{"status": "online", ...}`

Остановите сервер: Ctrl+C

---

## Часть 3: Настройка systemd service (автозапуск)

### 3.1 Создание systemd service файла

Вернитесь к root пользователю:

```bash
exit  # если вы под csvagent
```

Создайте service файл:

```bash
sudo nano /etc/systemd/system/csvagent.service
```

Вставьте следующее содержимое:

```ini
[Unit]
Description=CSV Analysis Agent API Server
After=network.target

[Service]
Type=simple
User=csvagent
Group=csvagent
WorkingDirectory=/home/csvagent/Claude_code
Environment="PATH=/home/csvagent/Claude_code/venv/bin"
ExecStart=/home/csvagent/Claude_code/venv/bin/python api_server.py
Restart=always
RestartSec=10

# Логирование
StandardOutput=append:/var/log/csvagent/access.log
StandardError=append:/var/log/csvagent/error.log

[Install]
WantedBy=multi-user.target
```

Сохраните файл.

### 3.2 Создание директории для логов

```bash
sudo mkdir -p /var/log/csvagent
sudo chown csvagent:csvagent /var/log/csvagent
```

### 3.3 Запуск и включение службы

```bash
# Перезагрузить systemd
sudo systemctl daemon-reload

# Запустить службу
sudo systemctl start csvagent

# Проверить статус
sudo systemctl status csvagent

# Включить автозапуск при загрузке системы
sudo systemctl enable csvagent
```

### 3.4 Проверка логов

```bash
# Просмотр логов
sudo tail -f /var/log/csvagent/access.log
sudo tail -f /var/log/csvagent/error.log

# Или через journalctl
sudo journalctl -u csvagent -f
```

---

## Часть 4: Настройка Nginx (reverse proxy)

### 4.1 Создание конфигурации Nginx

```bash
sudo nano /etc/nginx/sites-available/csvagent
```

Вставьте:

```nginx
server {
    listen 80;
    server_name your_domain.com;  # Замените на ваш домен или IP

    # Ограничение размера загружаемых файлов
    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Таймауты для долгих запросов
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }

    # Логи Nginx
    access_log /var/log/nginx/csvagent_access.log;
    error_log /var/log/nginx/csvagent_error.log;
}
```

### 4.2 Активация конфигурации

```bash
# Создать symlink
sudo ln -s /etc/nginx/sites-available/csvagent /etc/nginx/sites-enabled/

# Проверить конфигурацию
sudo nginx -t

# Перезапустить Nginx
sudo systemctl restart nginx

# Включить автозапуск
sudo systemctl enable nginx
```

### 4.3 Настройка Firewall

```bash
# Разрешить HTTP и HTTPS
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH

# Включить firewall
sudo ufw enable

# Проверить статус
sudo ufw status
```

---

## Часть 5: Настройка SSL (HTTPS) с Let's Encrypt

### 5.1 Установка Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 5.2 Получение SSL сертификата

```bash
# Замените your_domain.com на ваш домен
sudo certbot --nginx -d your_domain.com

# Следуйте инструкциям на экране
# Выберите опцию redirect HTTP -> HTTPS
```

### 5.3 Автоматическое обновление сертификата

```bash
# Проверить автоматическое обновление
sudo certbot renew --dry-run
```

Certbot автоматически добавит задачу в cron для обновления сертификатов.

---

## Часть 6: Проверка работы API

### 6.1 Проверка через curl

```bash
# Health check
curl http://your_domain.com/health

# Должно вернуть:
# {"status":"healthy","timestamp":"..."}
```

### 6.2 Тестовый запрос с файлом

Создайте тестовый CSV файл:

```bash
cat > test.csv << EOF
name,age,salary
Alice,30,50000
Bob,25,45000
Charlie,35,60000
EOF
```

Отправьте запрос:

```bash
curl -X POST "http://your_domain.com/api/analyze" \
  -F "file=@test.csv" \
  -F "query=Какая средняя зарплата?"
```

Должны получить JSON с результатом анализа.

---

## Часть 7: Мониторинг и обслуживание

### 7.1 Просмотр логов

```bash
# Логи приложения
sudo tail -f /var/log/csvagent/access.log
sudo tail -f /var/log/csvagent/error.log

# Логи Nginx
sudo tail -f /var/log/nginx/csvagent_access.log
sudo tail -f /var/log/nginx/csvagent_error.log

# Системные логи
sudo journalctl -u csvagent -f
```

### 7.2 Управление службой

```bash
# Статус
sudo systemctl status csvagent

# Остановка
sudo systemctl stop csvagent

# Запуск
sudo systemctl start csvagent

# Перезапуск
sudo systemctl restart csvagent

# Перезагрузка конфигурации
sudo systemctl reload csvagent
```

### 7.3 Обновление кода

```bash
# Переключиться на пользователя csvagent
sudo su - csvagent
cd ~/Claude_code

# Получить обновления
git pull

# Активировать venv
source venv/bin/activate

# Обновить зависимости (если изменились)
pip install -r requirements_api.txt

# Выйти из csvagent
exit

# Перезапустить службу
sudo systemctl restart csvagent
```

---

## Часть 8: Оптимизация для Production

### 8.1 Использование Gunicorn (рекомендуется)

Измените ExecStart в `/etc/systemd/system/csvagent.service`:

```ini
ExecStart=/home/csvagent/Claude_code/venv/bin/gunicorn api_server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600 \
    --access-logfile /var/log/csvagent/access.log \
    --error-logfile /var/log/csvagent/error.log
```

Перезапустите:

```bash
sudo systemctl daemon-reload
sudo systemctl restart csvagent
```

### 8.2 Настройка ротации логов

Создайте файл:

```bash
sudo nano /etc/logrotate.d/csvagent
```

Содержимое:

```
/var/log/csvagent/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 csvagent csvagent
    sharedscripts
    postrotate
        systemctl reload csvagent > /dev/null 2>&1 || true
    endscript
}
```

---

## Часть 9: Troubleshooting

### Проблема: Сервис не запускается

```bash
# Проверить логи
sudo journalctl -u csvagent -n 50

# Проверить права на файлы
ls -la /home/csvagent/Claude_code

# Проверить .env файл
sudo su - csvagent
cat ~/Claude_code/.env
```

### Проблема: Nginx показывает 502 Bad Gateway

```bash
# Проверить запущен ли API сервер
sudo systemctl status csvagent

# Проверить порт
sudo netstat -tlnp | grep 8000

# Проверить логи Nginx
sudo tail -f /var/log/nginx/csvagent_error.log
```

### Проблема: Ошибка загрузки больших файлов

Увеличьте лимиты в Nginx:

```nginx
# В /etc/nginx/sites-available/csvagent
client_max_body_size 500M;  # Увеличьте размер
```

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### Проблема: Медленная работа

```bash
# Увеличьте количество workers в gunicorn
# В /etc/systemd/system/csvagent.service
# --workers 8  (вместо 4)

# Добавьте больше RAM
# Или оптимизируйте код
```

---

## Часть 10: Безопасность

### 10.1 Ограничение доступа к API

Добавьте в Nginx basic auth:

```bash
# Установить утилиту
sudo apt install apache2-utils

# Создать пароль
sudo htpasswd -c /etc/nginx/.htpasswd apiuser
```

Обновите конфиг Nginx:

```nginx
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    proxy_pass http://127.0.0.1:8000;
    # ... остальное без изменений
}
```

### 10.2 Rate limiting в Nginx

Добавьте в `/etc/nginx/nginx.conf` в блок `http`:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
```

В конфиге сайта:

```nginx
location /api/ {
    limit_req zone=api_limit burst=20;
    # ... остальное
}
```

---

## Часть 11: Автоматический мониторинг

### 11.1 Установка простого health check

Создайте скрипт:

```bash
sudo nano /usr/local/bin/check_csvagent.sh
```

```bash
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ "$response" != "200" ]; then
    echo "API не отвечает! Код: $response"
    systemctl restart csvagent
    echo "$(date): API перезапущен" >> /var/log/csvagent/restart.log
fi
```

```bash
sudo chmod +x /usr/local/bin/check_csvagent.sh
```

Добавьте в crontab:

```bash
sudo crontab -e
```

Добавьте строку:

```
*/5 * * * * /usr/local/bin/check_csvagent.sh
```

---

## Резюме команд для быстрого деплоя

```bash
# 1. Обновление системы
sudo apt update && sudo apt upgrade -y

# 2. Установка пакетов
sudo apt install -y python3 python3-pip python3-venv git nginx

# 3. Клонирование репо
git clone https://github.com/your-repo/Claude_code.git
cd Claude_code

# 4. Виртуальное окружение
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_api.txt

# 5. Настройка .env
echo "OPENROUTER_API_KEY=your_key" > .env

# 6. Создание service
sudo cp csvagent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start csvagent
sudo systemctl enable csvagent

# 7. Настройка Nginx
sudo cp nginx_csvagent.conf /etc/nginx/sites-available/csvagent
sudo ln -s /etc/nginx/sites-available/csvagent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 8. SSL (если есть домен)
sudo certbot --nginx -d your_domain.com

# Готово!
```

---

## Полезные ссылки

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [Let's Encrypt](https://letsencrypt.org/)
- [UFW Firewall](https://help.ubuntu.com/community/UFW)

---

## Поддержка

При возникновении проблем:
1. Проверьте логи
2. Проверьте статус служб
3. Проверьте firewall
4. Создайте issue в репозитории

**API готов к работе!** 🚀
