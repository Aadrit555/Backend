import json
import os
from pathlib import Path
from pprint import pprint

from backend.dataset.chat_parser import ChatGPTParser

def create_synthetic_data(path: Path):
    data = [
        {
            "title": "Normal Conversation",
            "mapping": {
                "id1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello!"]},
                        "create_time": 100
                    }
                },
                "id2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi! How can I help?"]},
                        "create_time": 101
                    }
                }
            }
        },
        {
            "title": "Conversation with Empty Message",
            "mapping": {
                "id1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [""]},
                        "create_time": 100
                    }
                },
                "id2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": [""]},
                        "create_time": 101
                    }
                },
                "id3": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Wait, are you there?"]},
                        "create_time": 102
                    }
                }
            }
        },
        {
            "title": "Duplicated Conversation",
            "mapping": {
                "id1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello!"]},
                        "create_time": 100
                    }
                },
                "id2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi! How can I help?"]},
                        "create_time": 101
                    }
                }
            }
        },
        {
            "title": "Empty Conversation",
            "mapping": {
                "id1": {
                    "message": {
                        "author": {"role": "system"},
                        "content": {"content_type": "text", "parts": [""]},
                        "create_time": 100
                    }
                }
            }
        }
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    scratch_dir = Path("backend/scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = scratch_dir / "synthetic_conversations.json"
    output_path = scratch_dir / "sharegpt_dataset.jsonl"
    
    print("=== Creating synthetic conversations.json ===")
    create_synthetic_data(input_path)
    
    print(f"Input file created at: {input_path}")
    print("--- EXACT INPUT FILE CONTENT ---")
    with open(input_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("--------------------------------\n")
    
    parser = ChatGPTParser()
    print("=== Running Pipeline ===")
    raw_count, clean_count, removed_log = parser.parse(input_path, output_path)
    
    print(f"\nStats:")
    print(f"Total Raw Conversations: {raw_count}")
    print(f"Final Clean Conversations: {clean_count}")
    
    print("\nRemoved Lines / Audit Log:")
    for log in removed_log:
        print(f" - {log}")
        
    print("\n--- EXACT OUTPUT FILE CONTENT (.jsonl) ---")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("------------------------------------------")

if __name__ == "__main__":
    main()
