from typing import Iterator
from langchain_core.messages import BaseMessageChunk


def print_llm_stream(
    stream: Iterator[BaseMessageChunk], print_reasoning=True, print_response=True
):
    response = ""
    print("THINKING...\n")
    for i in stream:
        if "reasoning_content" in i.additional_kwargs.keys() and print_reasoning:
            print(i.additional_kwargs["reasoning_content"], end="")
        else:
            if len(response) == 0 and print_reasoning:
                print("\n\nTHINKING END")
            response += str(i.content)
            if print_response:
                print(i.content, end="")
    print("")
    return response


def get_tag(s: str, tag: str):
    find_str_open = "<" + tag + ">"
    find_str_close = "</" + tag + ">"
    open_idx = s.find(find_str_open) + len(tag) + 2
    close_idx = s.find(find_str_close)
    return s[open_idx:close_idx]


def get_bool_tag(s: str, tag: str):
    find_str = "<" + tag + "/>"
    idx = s.find(find_str)
    return idx != -1
