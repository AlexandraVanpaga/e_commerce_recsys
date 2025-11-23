from airflow.providers.telegram.hooks.telegram import TelegramHook

def send_telegram_success_message(context):
    hook = TelegramHook(token='8351974589:AAGEnSiHQTpS98r1O2Xo_FG4aMp66Yqraao', chat_id='-4856685146')
    dag = context['dag'].dag_id
    run_id = context['run_id']
    message = f'Исполнение DAG {dag} с id={run_id} прошло успешно!'
    hook.send_message({'chat_id': '-4856685146', 'text': message})

def send_telegram_failure_message(context):
    hook = TelegramHook(token='8351974589:AAGEnSiHQTpS98r1O2Xo_FG4aMp66Yqraao', chat_id='-4856685146')
    run_id = context['run_id']
    task_instance_key_str = context['task_instance_key_str']
    message = (
        f'Ошибка выполнения DAG!\n'
        f'Run ID: {run_id}\n'
        f'Task: {task_instance_key_str}'
    )
    hook.send_message({'chat_id': '-4856685146', 'text': message})