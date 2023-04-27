import random
from paho.mqtt import client as mqtt_client


def connect_mqtt(broker_ip, port) -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client_id = f'python-mqtt-{random.randint(0, 100)}'
    client = mqtt_client.Client(client_id)
    #client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker_ip, port)
    return client


def subscribe(client: mqtt_client, topic, message_callback_func):
    """
    :param client: the mqtt client created by function connect_mqtt
    :param topic: the topic to whic want to subscribe
    :param on_message_func: the function that we want to execute when a message is received
            the function must have format def func(client, userdata, msg)
    """
    client.subscribe(topic)
    client.message_callback_add(topic, message_callback_func)


def publish(client: mqtt_client, topic, message):
    result = client.publish(topic, message)
    status = result[0]
    if status != 0:
        raise Exception(f"Failed to send message to topic {topic}")


def show_message(client, userdata, msg):
    print(f"Received {msg.payload.decode()}")


if __name__ == "__main__":
    client = connect_mqtt('localhost', 1883)
    #subscribe(client, "pico", show_message)
    #subscribe(client, "lol", show_message2)

    from utils import all_transforms

    def process_message(client, userdata, msg):
        t = all_transforms(msg.payload.decode())
        print(f"Received {t}")

    subscribe(client, "pico", process_message)

    client.loop_forever()