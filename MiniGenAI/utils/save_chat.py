import json
import os


def save_chat(messages):

    os.makedirs(
        "chats",
        exist_ok=True
    )

    with open(
        "chats/history.json",
        "w"
    ) as f:

        json.dump(
            messages,
            f,
            indent=4
        )