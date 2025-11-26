from airflow.plugins_manager import AirflowPlugin


def send_telegram_success_message(context):
    from airflow.providers.telegram.hooks.telegram import TelegramHook
    hook = TelegramHook(token='8351974589:AAGEnSiHQTpS98r1O2Xo_FG4aMp66Yqraao', chat_id='-4856685146')
    dag = context['dag'].dag_id
    run_id = context['run_id']
    message = f'DAG {dag} (ID: {run_id}) выполнен успешно!'
    hook.send_message({'text': message})

def send_telegram_failure_message(context):
    from airflow.providers.telegram.hooks.telegram import TelegramHook
    hook = TelegramHook(token='8351974589:AAGEnSiHQTpS98r1O2Xo_FG4aMp66Yqraao', chat_id='-4856685146')
    run_id = context['run_id']
    task = context['task_instance_key_str']
    message = f'Ошибка! Run: {run_id}, Task: {task}'
    hook.send_message({'text': message})